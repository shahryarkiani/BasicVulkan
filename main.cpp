#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> deviceExtensions = {
    vk::KHRSwapchainExtensionName,
    vk::EXTMeshShaderExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRShaderFloatControlsExtensionName,
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Camera {
  glm::vec3 position;
  float pitch;
  float yaw;
  float vfov;

  Camera() : position({0, 0, 0}), pitch(0), yaw(0), vfov(0) {}

  Camera(glm::vec3 initPosition, float pitch, float yaw, float vfov)
      : position(initPosition), pitch(pitch), yaw(yaw), vfov(vfov) {}

  [[nodiscard]] glm::vec3 getForwardVector() const {
    glm::vec3 direction;
    direction.x = cos(pitch) * cos(yaw);
    direction.y = sin(pitch);
    direction.z = cos(pitch) * sin(yaw);
    return glm::normalize(direction);
  }
  [[nodiscard]] glm::mat4 getViewTransform() const {
    // Compute target and view matrix
    return glm::lookAt(position, position + getForwardVector(), {0, 1, 0});
  }
  void processMovement(float forward, float horizontal, float vertical, float yaw_change,
                       float pitch_change) {
    glm::vec3 direction = glm::normalize(getForwardVector());
    glm::vec3 right = glm::normalize(glm::cross(direction, {0, 1, 0}));
    glm::vec3 up = {0, 1, 0};

    position += direction * forward;
    position += right * horizontal;
    position += vertical * up;


    pitch += pitch_change;
    pitch = std::clamp(pitch, -1.55f, 1.55f);

    yaw += yaw_change;
  }
};

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription;
    bindingDescription.setBinding(0);
    bindingDescription.setStride(sizeof(Vertex));
    bindingDescription.setInputRate(vk::VertexInputRate::eVertex);

    return bindingDescription;
  }

  static std::array<vk::VertexInputAttributeDescription, 3>
  getAttributeDescriptions() {
    vk::VertexInputAttributeDescription posDescription;
    posDescription.setBinding(0);
    posDescription.setLocation(0);
    posDescription.setFormat(vk::Format::eR32G32B32Sfloat);
    posDescription.setOffset(offsetof(Vertex, pos));

    vk::VertexInputAttributeDescription colorDescription;
    colorDescription.setBinding(0);
    colorDescription.setLocation(1);
    colorDescription.setFormat(vk::Format::eR32G32B32Sfloat);
    colorDescription.setOffset(offsetof(Vertex, color));

    vk::VertexInputAttributeDescription texCoordDescription;
    texCoordDescription.binding = 0;
    texCoordDescription.location = 2;
    texCoordDescription.format = vk::Format::eR32G32Sfloat;
    texCoordDescription.offset = offsetof(Vertex, texCoord);

    return {posDescription, colorDescription, texCoordDescription};
  }
};

struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  GLFWwindow *window = nullptr;

  vk::Instance instance;

  vk::PhysicalDevice physicalDevice;
  vk::Device device;

  vk::SwapchainKHR swapChain;
  std::vector<vk::Image> swapChainImages;
  vk::Format swapChainImageFormat = vk::Format::eUndefined;
  vk::Extent2D swapChainExtent;
  std::vector<vk::Framebuffer> swapChainFramebuffers;

  vk::RenderPass renderPass;

  vk::DescriptorSetLayout descriptorSetLayout;
  vk::PipelineLayout pipelineLayout;
  vk::PipelineLayout meshPipelineLayout;

  vk::CommandPool commandPool;
  std::vector<vk::CommandBuffer> commandBuffers;
  vk::Pipeline graphicsPipeline;
  vk::Pipeline meshGraphicsPipeline;

  std::vector<vk::ImageView> swapChainImageViews;

  vk::Queue graphicsQueue;
  vk::Queue presentQueue;

  vk::SurfaceKHR surface;

  std::vector<vk::Semaphore> imageAvailableSemaphores;
  std::vector<vk::Semaphore> renderFinishedSemaphores;
  std::vector<vk::Fence> inFlightFences;

  vk::Buffer vertexBuffer;
  vk::DeviceMemory vertexBufferMemory;
  vk::Buffer indexBuffer;
  vk::DeviceMemory indexBufferMemory;

  vk::Sampler textureSampler;
  vk::ImageView textureImageView;
  vk::Image textureImage;
  vk::DeviceMemory textureImageMemory;

  vk::Image depthImage;
  vk::DeviceMemory depthImageMemory;
  vk::ImageView depthImageView;

  // This should be big enough to store MAX_FRAMES_IN_FLIGHTS count of
  // uniform buffer objects
  std::vector<vk::Buffer> uniformBuffers;
  vk::DeviceMemory uniformBufferMemory;
  UniformBufferObject *uniformBufferMapped = nullptr;

  vk::DescriptorPool descriptorPool;
  std::vector<vk::DescriptorSet> descriptorSets;

  Camera camera = Camera{
      {2.0f, 2.0f, 2.0f}, glm::radians(-30.f), glm::radians(-110.0f), 45};
  std::chrono::steady_clock::time_point prevTime;
  float sensitivity = 0.001;
  double lastX = 0.0, lastY = 0.0;
  bool firstMouse = true;

  const std::vector<Vertex> vertices = {
      {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
      {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
      {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
      {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

      {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
      {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
      {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
      {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}};

  const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7};

  uint32_t currentFrameIndex = 0;
  bool framebufferResized = false;

  static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan C++", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    const auto app = static_cast<HelloTriangleApplication *>(
        glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void initVulkan() {
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createMeshGraphicsPipeline();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSet();
    createCommandBuffers();
    createSyncObjects();
  }

  void createDepthResources() {
    vk::Format depthFormat = findDepthFormat();

    std::tie(depthImage, depthImageMemory) =
        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
                    vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eDepthStencilAttachment,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);

    depthImageView = createImageView(depthImage, depthFormat,
                                     vk::ImageAspectFlagBits::eDepth);
  }

  bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint ||
           format == vk::Format::eD24UnormS8Uint;
  }

  vk::Format findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features) {
    for (vk::Format format : candidates) {
      vk::FormatProperties properties =
          physicalDevice.getFormatProperties(format);

      if (tiling == vk::ImageTiling::eLinear &&
          (properties.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == vk::ImageTiling::eOptimal &&
                 (properties.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    throw std::runtime_error("Couldn't find supported format");
  }

  void createTextureSampler() {
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);
    samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAnisotropyEnable(vk::True);
    samplerInfo.setMaxAnisotropy(
        physicalDevice.getProperties().limits.maxSamplerAnisotropy);
    samplerInfo.setBorderColor(vk::BorderColor::eIntOpaqueBlack);
    samplerInfo.setUnnormalizedCoordinates(vk::False);
    samplerInfo.setCompareEnable(vk::False);
    samplerInfo.setCompareOp(vk::CompareOp::eAlways);
    samplerInfo.setMipmapMode(vk::SamplerMipmapMode::eLinear);
    samplerInfo.setMinLod(0.0);
    samplerInfo.setMaxLod(0.0);
    samplerInfo.setMipLodBias(0.0);

    textureSampler = device.createSampler(samplerInfo);
  }

  void createTextureImageView() {
    textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb,
                                       vk::ImageAspectFlagBits::eColor);
  }

  vk::ImageView createImageView(vk::Image image, vk::Format format,
                                vk::ImageAspectFlags aspectMask) const {
    vk::ImageViewCreateInfo viewInfo;
    viewInfo.setImage(image);
    viewInfo.setFormat(format);
    viewInfo.setViewType(vk::ImageViewType::e2D);
    vk::ImageSubresourceRange range;
    range.setAspectMask(aspectMask);
    range.setBaseMipLevel(0);
    range.setLevelCount(1);
    range.setBaseArrayLayer(0);
    range.setLayerCount(1);
    viewInfo.setSubresourceRange(range);

    return device.createImageView(viewInfo);
  }

  void createTextureImage() {
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight,
                                &texChannels, STBI_rgb_alpha);
    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    const vk::DeviceSize imageSize = texWidth * texHeight * 4;

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    void *data;
    const auto result =
        device.mapMemory(stagingBufferMemory, 0, imageSize,
                         static_cast<vk::MemoryMapFlagBits>(0), &data);
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to map memory for image staging buffer");
    }
    std::memcpy(data, pixels, imageSize);
    device.unmapMemory(stagingBufferMemory);
    stbi_image_free(pixels);

    std::tie(textureImage, textureImageMemory) = createImage(
        texWidth, texHeight, vk::Format::eR8G8B8A8Srgb,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
    transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb,
                          vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal);
    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
  }

  std::pair<vk::Image, vk::DeviceMemory> createImage(
      uint32_t width, uint32_t height, vk::Format format,
      vk::ImageTiling tiling, vk::ImageUsageFlags usage,
      vk::MemoryPropertyFlags properties) {
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({static_cast<uint32_t>(width),
                         static_cast<uint32_t>(height), /*depth =*/1});
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setTiling(tiling);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    imageInfo.setUsage(usage);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);

    const vk::Image image = device.createImage(imageInfo);

    const vk::MemoryRequirements memRequirements =
        device.getImageMemoryRequirements(image);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(
        findMemoryType(memRequirements.memoryTypeBits, properties));

    const vk::DeviceMemory imageMemory = device.allocateMemory(allocInfo);

    device.bindImageMemory(image, imageMemory, 0);

    return {image, imageMemory};
  }

  void transitionImageLayout(vk::Image image, vk::Format format,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout) {
    const auto commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier;
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrier.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrier.setImage(image);
    vk::ImageSubresourceRange range;
    range.setAspectMask(vk::ImageAspectFlagBits::eColor);
    range.setBaseMipLevel(0);
    range.setLevelCount(1);
    range.setBaseArrayLayer(0);
    range.setLayerCount(1);
    barrier.setSubresourceRange(range);

    vk::PipelineStageFlags srcStage;
    vk::PipelineStageFlags dstStage;

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.setSrcAccessMask(vk::AccessFlagBits::eNone);
      barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

      srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
      dstStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
      barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

      srcStage = vk::PipelineStageFlagBits::eTransfer;
      dstStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::runtime_error("Unsupported image layout transition");
    }

    commandBuffer.pipelineBarrier(srcStage, dstStage, vk::DependencyFlags(0),
                                  {}, {}, barrier);

    endSingleTimeCommands(commandBuffer);
  }

  void copyBufferToImage(const vk::Buffer buffer, const vk::Image image,
                         const uint32_t width, const uint32_t height) const {
    const auto commandBuffer = beginSingleTimeCommands();

    vk::BufferImageCopy region;
    region.setBufferOffset(0);
    region.setBufferRowLength(0);
    region.setBufferImageHeight(0);

    vk::ImageSubresourceLayers imageSubresourceLayers;
    imageSubresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    imageSubresourceLayers.setMipLevel(0);
    imageSubresourceLayers.setBaseArrayLayer(0);
    imageSubresourceLayers.setLayerCount(1);

    region.setImageSubresource(imageSubresourceLayers);
    region.setImageOffset(vk::Offset3D(0, 0, 0));
    region.setImageExtent(vk::Extent3D(width, height, 1));

    commandBuffer.copyBufferToImage(
        buffer, image, vk::ImageLayout::eTransferDstOptimal, region);

    endSingleTimeCommands(commandBuffer);
  }

  void createDescriptorSet() {
    vk::DescriptorSetAllocateInfo allocInfo;
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 descriptorSetLayout);
    allocInfo.setSetLayouts(layouts);
    allocInfo.setDescriptorPool(descriptorPool);

    descriptorSets = device.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo uboBufferInfo;
      uboBufferInfo.buffer = uniformBuffers[i];
      uboBufferInfo.offset = 0;
      uboBufferInfo.range = sizeof(UniformBufferObject);

      vk::WriteDescriptorSet uboDescriptorWrite;
      uboDescriptorWrite.setDstSet(descriptorSets[i]);
      uboDescriptorWrite.setDstBinding(0);
      uboDescriptorWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
      uboDescriptorWrite.setDescriptorCount(1);
      uboDescriptorWrite.setDstArrayElement(0);
      uboDescriptorWrite.setBufferInfo(uboBufferInfo);

      vk::DescriptorImageInfo imageInfo;
      imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
      imageInfo.setImageView(textureImageView);
      imageInfo.setSampler(textureSampler);

      vk::WriteDescriptorSet samplerDescriptorWrite;
      samplerDescriptorWrite.setDstSet(descriptorSets[i]);
      samplerDescriptorWrite.setDstBinding(1);
      samplerDescriptorWrite.setDescriptorType(
          vk::DescriptorType::eCombinedImageSampler);
      samplerDescriptorWrite.setDescriptorCount(1);
      samplerDescriptorWrite.setDstArrayElement(0);
      samplerDescriptorWrite.setImageInfo(imageInfo);

      device.updateDescriptorSets({uboDescriptorWrite, samplerDescriptorWrite},
                                  {});
    }
  }

  void createDescriptorPool() {
    vk::DescriptorPoolSize uboPoolSize;
    uboPoolSize.setType(vk::DescriptorType::eUniformBuffer);
    uboPoolSize.setDescriptorCount(MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolSize samplerPoolSize;
    samplerPoolSize.setType(vk::DescriptorType::eCombinedImageSampler);
    samplerPoolSize.setDescriptorCount(MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolCreateInfo poolInfo;
    std::vector<vk::DescriptorPoolSize> poolSizes{uboPoolSize, samplerPoolSize};
    poolInfo.setPoolSizes(poolSizes);
    poolInfo.setMaxSets(MAX_FRAMES_IN_FLIGHT);

    descriptorPool = device.createDescriptorPool(poolInfo);
  }

  void createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding samplerLayoutBinding;
    samplerLayoutBinding.setBinding(1);
    samplerLayoutBinding.setDescriptorCount(1);
    samplerLayoutBinding.setDescriptorType(
        vk::DescriptorType::eCombinedImageSampler);
    samplerLayoutBinding.setStageFlags(vk::ShaderStageFlagBits::eFragment);

    vk::DescriptorSetLayoutBinding uboLayoutBinding;
    uboLayoutBinding.setBinding(0);
    uboLayoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
    uboLayoutBinding.setDescriptorCount(1);
    uboLayoutBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex |
                                   vk::ShaderStageFlagBits::eMeshEXT |
                                   vk::ShaderStageFlagBits::eTaskEXT);

    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    std::vector<vk::DescriptorSetLayoutBinding> bindings{uboLayoutBinding,
                                                         samplerLayoutBinding};
    layoutInfo.setBindings(bindings);

    descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
  }

  void updateCamera() {
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float deltaTime =
        std::chrono::duration<float, std::chrono::milliseconds::period>(
            currentTime - prevTime)
            .count();

    float forward = 0.0;
    float horizontal = 0.0;
    float vertical = 0.0;
    float pitchChange = 0.0;
    float yawChange = 0.0;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) forward += 1.0;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) forward -= 1.0;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) horizontal += 1.0;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) horizontal -= 1.0;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) vertical += 1.0; 
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) vertical -= 1.0;
        
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (firstMouse) {
      lastX = xpos;
      lastY = ypos;
      firstMouse = false;
    }

    double xoffset = xpos - lastX;
    double yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    yawChange = static_cast<float>(xoffset) * sensitivity;
    pitchChange = static_cast<float>(yoffset) * sensitivity;
    camera.processMovement(forward * deltaTime * 100,
                           horizontal * deltaTime * 100, vertical * deltaTime * 100, yawChange,
                           pitchChange);
  }

  void updateUniformBuffer(const uint32_t frameIdx) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float timeDiff =
        std::chrono::duration<float, std::chrono::seconds::period>(currentTime -
                                                                   startTime)
            .count();

    UniformBufferObject ubo{};

    // Rotate identity matrix about the Z axis based on time
    ubo.model = glm::rotate(glm::mat4(1.0f), timeDiff * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.view = camera.getViewTransform();

    ubo.proj = glm::perspective(
        glm::radians(camera.vfov),
        swapChainExtent.width / static_cast<float>(swapChainExtent.height),
        0.1f, 500.0f);
    ubo.proj[1][1] *= -1;

    std::memcpy(&uniformBufferMapped[frameIdx], &ubo, sizeof(ubo));
  }

  void createUniformBuffers() {
    constexpr vk::DeviceSize bufferListSize =
        sizeof(UniformBufferObject) * MAX_FRAMES_IN_FLIGHT;
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    vk::BufferCreateInfo bufferListInfo;
    bufferListInfo.setSize(bufferListSize);
    bufferListInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
    bufferListInfo.setSharingMode(vk::SharingMode::eExclusive);

    vk::Buffer tempBuffer = device.createBuffer(bufferListInfo);

    const vk::MemoryRequirements memRequirements =
        device.getBufferMemoryRequirements(tempBuffer);

    device.destroyBuffer(tempBuffer);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(
        findMemoryType(memRequirements.memoryTypeBits,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent));

    uniformBufferMemory = device.allocateMemory(allocInfo);

    const auto result =
        device.mapMemory(uniformBufferMemory, 0, bufferListSize,
                         static_cast<vk::MemoryMapFlagBits>(0),
                         reinterpret_cast<void **>(&uniformBufferMapped));
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to map uniform buffer memory");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::BufferCreateInfo bufferInfo;
      vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
      bufferInfo.setSize(bufferSize);
      bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
      bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
      uniformBuffers[i] = device.createBuffer(bufferInfo);
      device.bindBufferMemory(uniformBuffers[i], uniformBufferMemory,
                              i * sizeof(UniformBufferObject));
    }
  }

  void createIndexBuffer() {
    const vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data;
    const auto result =
        device.mapMemory(stagingBufferMemory, 0, bufferSize,
                         static_cast<vk::MemoryMapFlagBits>(0), &data);
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to map staging buffer memory");
    }
    std::memcpy(data, indices.data(), bufferSize);
    device.unmapMemory(stagingBufferMemory);

    std::tie(indexBuffer, indexBufferMemory) =
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eIndexBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
  }

  void createVertexBuffer() {
    const vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    std::tie(vertexBuffer, vertexBufferMemory) =
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eVertexBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    void *data;
    const auto result =
        device.mapMemory(stagingBufferMemory, 0, bufferSize,
                         static_cast<vk::MemoryMapFlagBits>(0), &data);
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to map staging buffer memory");
    }
    std::memcpy(data, vertices.data(), bufferSize);
    device.unmapMemory(stagingBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
  }

  [[nodiscard]] std::pair<vk::Buffer, vk::DeviceMemory> createBuffer(
      vk::DeviceSize size, vk::BufferUsageFlags usage,
      vk::MemoryPropertyFlags memProperties) const {
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(usage);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

    vk::Buffer buffer = device.createBuffer(bufferInfo);

    const vk::MemoryRequirements memRequirements =
        device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(
        findMemoryType(memRequirements.memoryTypeBits, memProperties));

    vk::DeviceMemory bufferMemory = device.allocateMemory(allocInfo);

    // bind buffer to memory, with offset of 0, because it's not being shared
    device.bindBufferMemory(buffer, bufferMemory, 0);

    return {buffer, bufferMemory};
  }

  vk::CommandBuffer beginSingleTimeCommands() const {
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool);
    allocInfo.setCommandBufferCount(1);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);

    vk::CommandBuffer commandBuffer =
        device.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    commandBuffer.begin(beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(vk::CommandBuffer commandBuffer) const {
    commandBuffer.end();

    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(commandBuffer);

    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();  // This is fine to do, not in the rendering loop

    device.freeCommandBuffers(commandPool, commandBuffer);
  }

  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                  vk::DeviceSize size) const {
    auto commandBuffer = beginSingleTimeCommands();

    vk::BufferCopy copyRegion;
    copyRegion.setSrcOffset(0);
    copyRegion.setDstOffset(0);
    copyRegion.size = size;

    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

    endSingleTimeCommands(commandBuffer);
  }

  [[nodiscard]] uint32_t findMemoryType(
      const uint32_t typeFilter,
      const vk::MemoryPropertyFlags properties) const {
    const vk::PhysicalDeviceMemoryProperties memProperties =
        physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if (typeFilter & 1 << i && (memProperties.memoryTypes[i].propertyFlags &
                                  properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("Unable to find suitable memory type");
  }

  void createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(swapChainImages.size());
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    vk::FenceCreateInfo fenceInfo;
    fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled);

    constexpr vk::SemaphoreCreateInfo semaphoreInfo;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
      inFlightFences[i] = device.createFence(fenceInfo);
    }
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
    }
  }

  void createCommandBuffers() {
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool);
    allocInfo.setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);

    commandBuffers = device.allocateCommandBuffers(allocInfo);
  }

  void recordCommandBuffer(const vk::CommandBuffer commandBuffer,
                           const uint32_t imageIndex) const {
    constexpr vk::CommandBufferBeginInfo beginInfo{};
    commandBuffer.begin(beginInfo);

    vk::RenderPassBeginInfo renderPassInfo{};
    renderPassInfo.setRenderPass(renderPass);
    renderPassInfo.setFramebuffer(swapChainFramebuffers[imageIndex]);

    renderPassInfo.renderArea.offset = vk::Offset2D(0, 0);
    renderPassInfo.renderArea.extent = swapChainExtent;

    vk::ClearValue clearColor{vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)};
    vk::ClearValue depthStencil{vk::ClearDepthStencilValue(1.0f, 0)};
    vk::ClearValue clearValues[] = {clearColor, depthStencil};
    renderPassInfo.setClearValues(clearValues);

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               graphicsPipeline);

    vk::Viewport viewport;
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    commandBuffer.setViewport(0, 1, &viewport);

    vk::Rect2D scissor;
    scissor.offset = vk::Offset2D(0, 0);
    scissor.extent = swapChainExtent;
    commandBuffer.setScissor(0, 1, &scissor);

    commandBuffer.bindVertexBuffers(0, vertexBuffer, {0});
    commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                     pipelineLayout, 0,
                                     descriptorSets[currentFrameIndex], {});



    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0,

                              0);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, meshGraphicsPipeline);

    PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT =
        (PFN_vkCmdDrawMeshTasksEXT) vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksEXT");

    vkCmdDrawMeshTasksEXT(commandBuffer, 1, 1, 1);

    commandBuffer.endRenderPass();
    commandBuffer.end();
  }

  void createCommandPool() {
    auto [graphicsFamily, presentFamily] = findQueueFamilies(physicalDevice);

    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(graphicsFamily.value());

    commandPool = device.createCommandPool(commandPoolInfo);
  }

  void createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
      std::array<vk::ImageView, 2> imageViews{swapChainImageViews[i],
                                              depthImageView};

      vk::FramebufferCreateInfo framebufferInfo;
      framebufferInfo.setRenderPass(renderPass);
      framebufferInfo.setAttachments(imageViews);
      framebufferInfo.setWidth(swapChainExtent.width);
      framebufferInfo.setHeight(swapChainExtent.height);
      framebufferInfo.setLayers(1);

      swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
    }
  }

  void createRenderPass() {
    vk::AttachmentDescription depthAttachment;
    depthAttachment.setFormat(findDepthFormat());
    depthAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
    depthAttachment.setStoreOp(vk::AttachmentStoreOp::eDontCare);
    depthAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
    depthAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
    depthAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
    depthAttachment.setFinalLayout(
        vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::AttachmentReference depthAttachmentRef;
    depthAttachmentRef.setAttachment(1);
    depthAttachmentRef.setLayout(
        vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::AttachmentDescription colorAttachment;
    colorAttachment.setFormat(swapChainImageFormat);
    colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
    colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);

    colorAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
    colorAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);

    colorAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
    colorAttachment.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    vk::AttachmentReference colorAttachmentRef;
    colorAttachmentRef.setAttachment(0);
    colorAttachmentRef.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpass;
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
    subpass.setColorAttachments(colorAttachmentRef);
    subpass.setPDepthStencilAttachment(&depthAttachmentRef);

    std::array<vk::AttachmentDescription, 2> attachments{colorAttachment,
                                                         depthAttachment};
    vk::RenderPassCreateInfo renderPassInfo;
    renderPassInfo.setAttachments(attachments);
    renderPassInfo.setSubpasses(subpass);

    vk::SubpassDependency dependency;
    dependency.setSrcSubpass(vk::SubpassExternal);
    dependency.setSrcStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput |
        vk::PipelineStageFlagBits::eLateFragmentTests);
    dependency.setSrcAccessMask(
        vk::AccessFlagBits::eDepthStencilAttachmentWrite);

    dependency.setDstSubpass(0);
    dependency.setDstStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput |
        vk::PipelineStageFlagBits::eEarlyFragmentTests);
    dependency.setDstAccessMask(
        vk::AccessFlagBits::eColorAttachmentWrite |
        vk::AccessFlagBits::eDepthStencilAttachmentRead);

    renderPassInfo.setDependencies(dependency);

    renderPass = device.createRenderPass(renderPassInfo);
  }

  void createMeshGraphicsPipeline() {
    auto taskShaderModule =
        createShaderModule(readFile("shaders/terrain.task.spv"));
    auto meshShaderModule =
        createShaderModule(readFile("shaders/terrain.mesh.spv"));
    auto fragShaderModule = createShaderModule(readFile("shaders/terrain.frag.spv"));

    vk::PipelineShaderStageCreateInfo taskShaderStageInfo(
        {}, vk::ShaderStageFlagBits::eTaskEXT, taskShaderModule, "main");
    vk::PipelineShaderStageCreateInfo meshShaderStageInfo(
        {}, vk::ShaderStageFlagBits::eMeshEXT, meshShaderModule, "main");

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo(
        {}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main");

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
        taskShaderStageInfo, meshShaderStageInfo, fragShaderStageInfo};

    std::vector dynamicStates = {vk::DynamicState::eViewport,
                                 vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState;
    dynamicState.setDynamicStates(dynamicStates);

    vk::PipelineViewportStateCreateInfo viewportStateCreateInfo({}, 1, {}, 1,
                                                                {});

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, vk::False, vk::False, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
        vk::False);
    rasterizer.setLineWidth(1.0f);

    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.setDepthTestEnable(vk::True);
    depthStencil.setDepthWriteEnable(vk::True);
    depthStencil.setDepthCompareOp(vk::CompareOp::eLess);
    depthStencil.setDepthBoundsTestEnable(vk::False);
    depthStencil.setStencilTestEnable(vk::False);

    // Default Initialize, we're not using multisampling
    vk::PipelineMultisampleStateCreateInfo multisampling;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.blendEnable = vk::False;
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;

    vk::PipelineColorBlendStateCreateInfo colorBlending;
    colorBlending.logicOpEnable = vk::False;
    colorBlending.setAttachments(colorBlendAttachment);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descriptorSetLayout);
    meshPipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

    vk::GraphicsPipelineCreateInfo pipelineInfo;
    pipelineInfo.setStages(shaderStages);
    pipelineInfo.setPViewportState(&viewportStateCreateInfo);
    pipelineInfo.setPRasterizationState(&rasterizer);
    pipelineInfo.setPMultisampleState(&multisampling);
    pipelineInfo.setPColorBlendState(&colorBlending);
    pipelineInfo.setPDynamicState(&dynamicState);
    pipelineInfo.setPDepthStencilState(&depthStencil);

    pipelineInfo.setLayout(meshPipelineLayout);
    pipelineInfo.setRenderPass(renderPass);
    pipelineInfo.setSubpass(0);

    vk::Result result;
    std::tie(result, meshGraphicsPipeline) =
        device.createGraphicsPipeline({}, pipelineInfo);

    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to create graphics pipeline\n");
    }
    device.destroyShaderModule(taskShaderModule);
    device.destroyShaderModule(meshShaderModule);
    device.destroyShaderModule(fragShaderModule);
  }

  void createGraphicsPipeline() {
    auto vertShaderModule = createShaderModule(readFile("shaders/vert.spv"));
    auto fragShaderModule = createShaderModule(readFile("shaders/frag.spv"));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo(
        {}, vk::ShaderStageFlags::BitsType::eVertex, vertShaderModule, "main");

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo(
        {}, vk::ShaderStageFlags::BitsType::eFragment, fragShaderModule,
        "main");

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
        vertShaderStageInfo, fragShaderStageInfo};

    std::vector dynamicStates = {vk::DynamicState::eViewport,
                                 vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState;
    dynamicState.setDynamicStates(dynamicStates);

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.setVertexBindingDescriptions(bindingDescription);
    vertexInputInfo.setVertexAttributeDescriptions(attributeDescriptions);

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo(
        {}, vk::PrimitiveTopology::eTriangleList, vk::False);

    vk::PipelineViewportStateCreateInfo viewportStateCreateInfo({}, 1, {}, 1,
                                                                {});

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, vk::False, vk::False, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
        vk::False);
    rasterizer.setLineWidth(1.0f);

    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.setDepthTestEnable(vk::True);
    depthStencil.setDepthWriteEnable(vk::True);
    depthStencil.setDepthCompareOp(vk::CompareOp::eLess);
    depthStencil.setDepthBoundsTestEnable(vk::False);
    depthStencil.setStencilTestEnable(vk::False);

    // Default Initialize, we're not using multisampling
    vk::PipelineMultisampleStateCreateInfo multisampling;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.blendEnable = vk::False;
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;

    vk::PipelineColorBlendStateCreateInfo colorBlending;
    colorBlending.logicOpEnable = vk::False;
    colorBlending.setAttachments(colorBlendAttachment);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descriptorSetLayout);
    pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

    vk::GraphicsPipelineCreateInfo pipelineInfo;
    pipelineInfo.setStages(shaderStages);
    pipelineInfo.setPVertexInputState(&vertexInputInfo);
    pipelineInfo.setPInputAssemblyState(&inputAssemblyInfo);
    pipelineInfo.setPViewportState(&viewportStateCreateInfo);
    pipelineInfo.setPRasterizationState(&rasterizer);
    pipelineInfo.setPMultisampleState(&multisampling);
    pipelineInfo.setPColorBlendState(&colorBlending);
    pipelineInfo.setPDynamicState(&dynamicState);
    pipelineInfo.setPDepthStencilState(&depthStencil);

    pipelineInfo.setLayout(pipelineLayout);
    pipelineInfo.setRenderPass(renderPass);
    pipelineInfo.setSubpass(0);

    vk::Result result;
    std::tie(result, graphicsPipeline) =
        device.createGraphicsPipeline({}, pipelineInfo);

    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to create graphics pipeline\n");
    }
    device.destroyShaderModule(vertShaderModule);
    device.destroyShaderModule(fragShaderModule);
  }

  [[nodiscard]] vk::ShaderModule createShaderModule(
      const std::vector<char> &shaderCode) const {
    const vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
        {}, shaderCode.size(),
        reinterpret_cast<const uint32_t *>(shaderCode.data()));

    return device.createShaderModule(shaderModuleCreateInfo);
  }

  void createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      swapChainImageViews[i] =
          createImageView(swapChainImages[i], swapChainImageFormat,
                          vk::ImageAspectFlagBits::eColor);
    }
  }

  void createSwapChain() {
    const auto [capabilities, formats, presentModes] =
        querySwapChainSupport(physicalDevice);

    const vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(formats);
    const vk::PresentModeKHR presentMode = chooseSwapPresentMode(presentModes);
    const vk::Extent2D extent = chooseSwapExtent(capabilities);

    uint32_t imageCount = capabilities.minImageCount + 1;

    if (capabilities.maxImageCount > 0 &&
        imageCount > capabilities.maxImageCount) {
      imageCount = capabilities.maxImageCount;
    }

    auto sharingMode = vk::SharingMode::eExclusive;
    const auto [graphicsFamily, presentFamily] =
        findQueueFamilies(physicalDevice);
    std::vector<uint32_t> queueIndicesList;

    if (graphicsFamily != presentFamily) {
      queueIndicesList = {graphicsFamily.value(), presentFamily.value()};
      sharingMode = vk::SharingMode::eConcurrent;
    }

    const vk::SwapchainCreateInfoKHR swapChainCreateInfo(
        {}, surface, imageCount, surfaceFormat.format, surfaceFormat.colorSpace,
        extent, 1, vk::ImageUsageFlagBits::eColorAttachment, sharingMode,
        queueIndicesList, capabilities.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, vk::True);
    swapChain = device.createSwapchainKHR(swapChainCreateInfo);

    swapChainExtent = extent;
    swapChainImageFormat = surfaceFormat.format;
    swapChainImages = device.getSwapchainImagesKHR(swapChain);
  }

  void recreateSwapChain() {
    device.waitIdle();

    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createDepthResources();
    createFramebuffers();
  }

  void cleanupSwapChain() const {
    for (const auto framebuffer : swapChainFramebuffers)
      device.destroy(framebuffer);

    for (const auto imageView : swapChainImageViews) device.destroy(imageView);
    device.destroyImageView(depthImageView);
    device.destroyImage(depthImage);
    device.freeMemory(depthImageMemory);
    device.destroySwapchainKHR(swapChain);
  }

  void createSurface() {
    VkSurfaceKHR tempSurface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &tempSurface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
    surface = tempSurface;
  }

  void mainLoop() {
    prevTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      prevTime = std::chrono::high_resolution_clock::now();
      updateCamera();
      drawFrame();
    }

    device.waitIdle();
  }

  void drawFrame() {
    auto result =
        device.waitForFences(inFlightFences[currentFrameIndex], vk::True,
                             std::numeric_limits<uint64_t>::max());

    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to wait for inFlightFence");
    }

    uint32_t nextImageIndex;
    std::tie(result, nextImageIndex) = device.acquireNextImageKHR(
        swapChain, std::numeric_limits<uint64_t>::max(),
        imageAvailableSemaphores[currentFrameIndex]);

    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    } else if (result != vk::Result::eSuccess &&
               result != vk::Result::eSuboptimalKHR) {
      throw std::runtime_error("Failed to acquire next image index");
    }
    // If we reset the fence, but return after recreating the swap chain
    // We have a deadlock scenario
    device.resetFences(inFlightFences[currentFrameIndex]);

    updateUniformBuffer(currentFrameIndex);

    commandBuffers[currentFrameIndex].reset();
    recordCommandBuffer(commandBuffers[currentFrameIndex], nextImageIndex);

    vk::SubmitInfo submitInfo;
    submitInfo.setWaitSemaphores(imageAvailableSemaphores[currentFrameIndex]);

    constexpr vk::PipelineStageFlags waitStages(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    submitInfo.setPWaitDstStageMask(&waitStages);

    submitInfo.setCommandBuffers(commandBuffers[currentFrameIndex]);

    submitInfo.setSignalSemaphores(renderFinishedSemaphores[nextImageIndex]);

    graphicsQueue.submit(submitInfo, inFlightFences[currentFrameIndex]);

    vk::PresentInfoKHR presentInfo;

    presentInfo.setWaitSemaphores(renderFinishedSemaphores[nextImageIndex]);
    presentInfo.setSwapchains(swapChain);
    presentInfo.setImageIndices(nextImageIndex);

    result = presentQueue.presentKHR(presentInfo);

    if (result == vk::Result::eErrorOutOfDateKHR || framebufferResized) {
      recreateSwapChain();
      framebufferResized = false;
    } else if (result != vk::Result::eSuccess &&
               result != vk::Result::eSuboptimalKHR) {
      throw std::runtime_error("Failed to acquire next image index");
    }

    currentFrameIndex++;
    currentFrameIndex %= MAX_FRAMES_IN_FLIGHT;
  }

  void cleanup() const {
    for (const auto semaphore : imageAvailableSemaphores)
      device.destroy(semaphore);
    for (const auto semaphore : renderFinishedSemaphores)
      device.destroy(semaphore);
    for (const auto fence : inFlightFences) device.destroy(fence);
    device.destroyCommandPool(commandPool);

    cleanupSwapChain();

    for (const auto uniformBuffer : uniformBuffers)
      device.destroyBuffer(uniformBuffer);
    device.destroySampler(textureSampler);
    device.destroyImageView(textureImageView);
    device.destroyImage(textureImage);
    device.freeMemory(textureImageMemory);
    device.freeMemory(uniformBufferMemory);
    device.destroyDescriptorPool(descriptorPool);
    device.destroyDescriptorSetLayout(descriptorSetLayout);
    device.destroyBuffer(indexBuffer);
    device.freeMemory(indexBufferMemory);
    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);
    device.destroyPipeline(graphicsPipeline);
    device.destroyPipeline(meshGraphicsPipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyPipelineLayout(meshPipelineLayout);
    device.destroyRenderPass(renderPass);

    device.destroy();
    if (instance) {
      instance.destroySurfaceKHR(surface);
      instance.destroy(nullptr);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
  }

  void createInstance() {
    if (!checkValidationLayerSupport()) {
      throw std::runtime_error(
          "Validation Enabled, but missing support for validation layers");
    }

    const std::vector<vk::ExtensionProperties> extensionProperties =
        vk::enumerateInstanceExtensionProperties();

    std::cout << "available instance extensions:\n";
    for (const auto &extension : extensionProperties) {
      std::cout << '\t' << extension.extensionName << '\n';
    }

    constexpr vk::ApplicationInfo appInfo(
        "Hello Triangle", VK_MAKE_VERSION(1, 0, 0), "No Engine",
        VK_MAKE_VERSION(1, 0, 0), vk::ApiVersion13);

    auto extensions = getRequiredExtensions();
    vk::InstanceCreateInfo createInfo;
    createInfo.setPApplicationInfo(&appInfo);
    createInfo.setPEnabledExtensionNames(extensions);

    if (enableValidationLayers) {
      createInfo.setPEnabledLayerNames(validationLayers);
    }

    instance = vk::createInstance(createInfo);
  }

  static std::vector<const char *> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> requiredExtensions(
        glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
      requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return requiredExtensions;
  }

  static bool checkValidationLayerSupport() {
    std::vector<vk::LayerProperties> availableLayers =
        vk::enumerateInstanceLayerProperties();

    for (const auto layerName : validationLayers) {
      bool layerFound = false;

      for (auto &layerProperty : availableLayers) {
        if (strcmp(layerName, layerProperty.layerName) != 0) {
          layerFound = true;
          break;
        }
      }
      if (!layerFound) return false;
    }

    return true;
  }

  void pickPhysicalDevice() {
    const std::vector<vk::PhysicalDevice> devices =
        instance.enumeratePhysicalDevices();

    if (devices.empty()) {
      throw std::runtime_error("No physical devices found");
    }

    for (const auto &candidateDevice : devices) {
      if (isDeviceSuitable(candidateDevice)) {
        physicalDevice = candidateDevice;
        return;
      }
    }

    throw std::runtime_error("No suitable device found!\n");
  }

  void createLogicalDevice() {
    auto [graphicsFamily, presentFamily] = findQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    std::set<uint32_t> uniqueQueueFamilies = {graphicsFamily.value(),
                                              presentFamily.value()};
    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1);
      queueCreateInfo.setPQueuePriorities(&queuePriority);

      queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceMeshShaderFeaturesEXT meshFeatures;
    meshFeatures.setMeshShader(vk::True);
    meshFeatures.setTaskShader(vk::True);

    vk::PhysicalDeviceMaintenance4Features main4features;
    main4features.setMaintenance4(vk::True);
    meshFeatures.setPNext(&main4features);

    vk::PhysicalDeviceFeatures2 deviceFeatures;
    deviceFeatures.features.setSamplerAnisotropy(vk::True);
    deviceFeatures.setPNext(&meshFeatures);

    vk::DeviceCreateInfo createInfo;
    createInfo.setQueueCreateInfos(queueCreateInfos);
    createInfo.setPNext(&deviceFeatures);

    createInfo.enabledExtensionCount = deviceExtensions.size();
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    device = physicalDevice.createDevice(createInfo);

    graphicsQueue = device.getQueue(graphicsFamily.value(), 0);
    presentQueue = device.getQueue(presentFamily.value(), 0);
  }

  [[nodiscard]] bool isDeviceSuitable(
      const vk::PhysicalDevice physDevice) const {
    const auto indices = findQueueFamilies(physDevice);

    const bool extensionsSupported = checkDeviceExtensionSupport(physDevice);

    const auto swapChainSupport = querySwapChainSupport(physDevice);

    vk::PhysicalDeviceFeatures2 supportedFeatures;
    vk::PhysicalDeviceMeshShaderFeaturesEXT meshFeatures;
    supportedFeatures.setPNext(&meshFeatures);

    vk::PhysicalDeviceMaintenance4Features main4features;
    meshFeatures.setPNext(&main4features);

    physDevice.getFeatures2(&supportedFeatures);

    return indices.isComplete() && swapChainSupport.isSuitable() &&
           extensionsSupported &&
           supportedFeatures.features.samplerAnisotropy &&
           meshFeatures.meshShader && main4features.maintenance4;
  }

  static bool checkDeviceExtensionSupport(
      const vk::PhysicalDevice physicalDevice) {
    const std::vector<vk::ExtensionProperties> availableExtensions =
        physicalDevice.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool isComplete() const {
      return graphicsFamily.has_value() && presentFamily.has_value();
    }
  };

  [[nodiscard]] QueueFamilyIndices findQueueFamilies(
      const vk::PhysicalDevice physDevice) const {
    QueueFamilyIndices indices;

    const std::vector<vk::QueueFamilyProperties> queueFamilies =
        physDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
      auto &queueFamily = queueFamilies[i];

      if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
        indices.graphicsFamily = i;
      }

      if (physDevice.getSurfaceSupportKHR(i, surface) == vk::True) {
        indices.presentFamily = i;
      }
    }

    return indices;
  }

  struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;

    [[nodiscard]] bool isSuitable() const {
      return !formats.empty() && !presentModes.empty();
    }
  };

  [[nodiscard]] SwapChainSupportDetails querySwapChainSupport(
      const vk::PhysicalDevice physDevice) const {
    SwapChainSupportDetails details;

    details.capabilities = physDevice.getSurfaceCapabilitiesKHR(surface);
    details.formats = physDevice.getSurfaceFormatsKHR(surface);
    details.presentModes = physDevice.getSurfacePresentModesKHR(surface);

    return details;
  }

  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    for (const auto &format : availableFormats) {
      if (format.format == vk::Format::eB8G8R8A8Srgb &&
          format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
        return format;
      }
    }

    return availableFormats[0];
  }

  static vk::PresentModeKHR chooseSwapPresentMode(
      const std::vector<vk::PresentModeKHR> &availablePresentModes) {
    for (const auto &presentMode : availablePresentModes) {
      if (presentMode == vk::PresentModeKHR::eMailbox) {
        return presentMode;
      }
    }

    return vk::PresentModeKHR::eFifo;
  }

  [[nodiscard]] vk::Extent2D chooseSwapExtent(
      const vk::SurfaceCapabilitiesKHR &capabilities) const {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    vk::Extent2D actualExtent(width, height);

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.maxImageExtent.height,
                   capabilities.maxImageExtent.height);

    return actualExtent;
  }

  static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file! " + filename);
    }

    const auto fileSize = file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
  }
};

int main() {
  try {
    HelloTriangleApplication app;
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
};
