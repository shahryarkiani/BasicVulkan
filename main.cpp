#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> deviceExtensions = {
    vk::KHRSwapchainExtensionName};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static vk::VertexInputBindingDescription getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription;
    bindingDescription.setBinding(0);
    bindingDescription.setStride(sizeof(Vertex));
    bindingDescription.setInputRate(vk::VertexInputRate::eVertex);

    return bindingDescription;
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    vk::VertexInputAttributeDescription posDescription;
    posDescription.setBinding(0);
    posDescription.setLocation(0);
    posDescription.setFormat(vk::Format::eR32G32Sfloat);
    posDescription.setOffset(offsetof(Vertex, pos));

    vk::VertexInputAttributeDescription colorDescription;
    colorDescription.setBinding(0);
    colorDescription.setLocation(1);
    colorDescription.setFormat(vk::Format::eR32G32B32Sfloat);
    colorDescription.setOffset(offsetof(Vertex, color));

    return {posDescription, colorDescription};
  }
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
  vk::PipelineLayout pipelineLayout;

  vk::CommandPool commandPool;
  std::vector<vk::CommandBuffer> commandBuffers;
  vk::Pipeline graphicsPipeline;

  std::vector<vk::ImageView> swapChainImageViews;

  vk::Queue graphicsQueue;
  vk::Queue presentQueue;

  vk::SurfaceKHR surface;

  std::vector<vk::Semaphore> imageAvailableSemaphores;
  std::vector<vk::Semaphore> renderFinishedSemaphores;
  std::vector<vk::Fence> inFlightFences;

  vk::Buffer vertexBuffer;
  vk::DeviceMemory vertexBufferMemory;

  const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
  };

  uint32_t currentFrameIndex = 0;
  bool framebufferResized = false;

  static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan C++", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
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
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createVertexBuffer();
    createCommandBuffers();
    createSyncObjects();
  }

  void createVertexBuffer() {
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(sizeof(vertices[0]) * vertices.size());
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eVertexBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

    vertexBuffer = device.createBuffer(bufferInfo);

    const vk::MemoryRequirements memRequirements =
        device.getBufferMemoryRequirements(vertexBuffer);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(
        findMemoryType(memRequirements.memoryTypeBits,
                       vk::MemoryPropertyFlagBits::eHostCoherent |
                           vk::MemoryPropertyFlagBits::eHostVisible));

    vertexBufferMemory = device.allocateMemory(allocInfo);

    // bind buffer to memory, with offset of 0, because it's not being shared
    device.bindBufferMemory(vertexBuffer, vertexBufferMemory, 0);

    void *data;
    auto result =
        device.mapMemory(vertexBufferMemory, 0, bufferInfo.size,
                         static_cast<vk::MemoryMapFlagBits>(0), &data);
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to map vertex buffer memory");
    }
    std::memcpy(data, vertices.data(), bufferInfo.size);
    device.unmapMemory(vertexBufferMemory);
  }

  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties) const {
    vk::PhysicalDeviceMemoryProperties memProperties =
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

    vk::ClearValue clearColor(vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f));
    renderPassInfo.setClearValues(clearColor);

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

    commandBuffer.draw(3, 1, 0, 0);

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
      vk::FramebufferCreateInfo framebufferInfo;
      framebufferInfo.setRenderPass(renderPass);
      framebufferInfo.setAttachments(swapChainImageViews[i]);
      framebufferInfo.setWidth(swapChainExtent.width);
      framebufferInfo.setHeight(swapChainExtent.height);
      framebufferInfo.setLayers(1);

      swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
    }
  }

  void createRenderPass() {
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

    vk::RenderPassCreateInfo renderPassInfo;
    renderPassInfo.setAttachments(colorAttachment);
    renderPassInfo.setSubpasses(subpass);

    vk::SubpassDependency dependency;
    dependency.setSrcSubpass(vk::SubpassExternal);
    dependency.setSrcStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    dependency.setSrcAccessMask(vk::AccessFlagBits::eNone);

    dependency.setDstSubpass(0);
    dependency.setDstStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    dependency.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

    renderPassInfo.setDependencies(dependency);

    renderPass = device.createRenderPass(renderPassInfo);
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
        vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise, vk::False);
    rasterizer.setLineWidth(1.0f);

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
    pipelineLayoutInfo.setSetLayouts({});
    pipelineLayoutInfo.setPushConstantRanges({});

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
      vk::ImageViewCreateInfo imageViewCreateInfo(
          {}, swapChainImages[i], vk::ImageViewType::e2D, swapChainImageFormat,
          vk::ComponentMapping(),
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                    1));

      swapChainImageViews[i] = device.createImageView(imageViewCreateInfo);
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
    createFramebuffers();
  }

  void cleanupSwapChain() const {
    for (const auto framebuffer : swapChainFramebuffers)
      device.destroy(framebuffer);

    for (const auto imageView : swapChainImageViews) device.destroy(imageView);

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
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
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

    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);
    device.destroyPipeline(graphicsPipeline);
    device.destroyPipelineLayout(pipelineLayout);
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
        VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0);

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

    vk::PhysicalDeviceFeatures deviceFeatures{};

    vk::DeviceCreateInfo createInfo;
    createInfo.setQueueCreateInfos(queueCreateInfos);
    createInfo.setPEnabledFeatures(&deviceFeatures);

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

    return indices.isComplete() && swapChainSupport.isSuitable() &&
           extensionsSupported;
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
      throw std::runtime_error("failed to open file!");
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
