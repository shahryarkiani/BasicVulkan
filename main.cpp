#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <fstream>
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

  vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
  vk::Device device;

  vk::SwapchainKHR swapChain;
  std::vector<vk::Image> swapChainImages;
  vk::Format swapChainImageFormat = vk::Format::eUndefined;
  vk::Extent2D swapChainExtent;

  vk::PipelineLayout pipelineLayout;


  std::vector<vk::ImageView> swapChainImageViews;

  vk::Queue graphicsQueue;
  vk::Queue presentQueue;

  vk::SurfaceKHR surface;

  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan C++", nullptr, nullptr);
  }

  void initVulkan() {
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createGraphicsPipeline();
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

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};

    // We are hardcoding the vertex inputs, so empty for everything
    vk::PipelineVertexInputStateCreateInfo({}, {}, {});

    vk::PipelineInputAssemblyStateCreateInfo(
        {}, vk::PrimitiveTopology::eTriangleList, vk::False);

    vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                          static_cast<float>(swapChainExtent.height), 0.0f,
                          1.0f);

    vk::Rect2D scissor({}, swapChainExtent);

    vk::PipelineViewportStateCreateInfo viewportStateCreateInfo({}, 1, {}, 1,
                                                                {});

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, vk::False, vk::False, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise, vk::False);

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
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SharingMode sharingMode = vk::SharingMode::eExclusive;
    QueueFamilyIndices queueIndices = findQueueFamilies(physicalDevice);
    std::vector<uint32_t> queueIndicesList;

    if (queueIndices.graphicsFamily != queueIndices.presentFamily) {
      queueIndicesList = {queueIndices.graphicsFamily.value(),
                          queueIndices.presentFamily.value()};
      sharingMode = vk::SharingMode::eConcurrent;
    }

    vk::SwapchainCreateInfoKHR swapChainCreateInfo(
        {}, surface, imageCount, surfaceFormat.format, surfaceFormat.colorSpace,
        extent, 1, vk::ImageUsageFlagBits::eColorAttachment, sharingMode,
        queueIndicesList, swapChainSupport.capabilities.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, vk::True);

    swapChain = device.createSwapchainKHR(swapChainCreateInfo);

    swapChainExtent = extent;
    swapChainImageFormat = surfaceFormat.format;
    swapChainImages = device.getSwapchainImagesKHR(swapChain);
  }

  void createSurface() {
    VkSurfaceKHR tempSurface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &tempSurface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
    surface = tempSurface;
  }

  void mainLoop() const {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanup() const {

    device.destroyPipelineLayout(pipelineLayout);

    for (const auto &imageView : swapChainImageViews) {
      device.destroyImageView(imageView);
    }

    device.destroySwapchainKHR(swapChain);
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

    std::vector<vk::ExtensionProperties> extensionProperties =
        vk::enumerateInstanceExtensionProperties();

    std::cout << "available instance extensions:\n";
    for (const auto &extension : extensionProperties) {
      std::cout << '\t' << extension.extensionName << '\n';
    }

    vk::ApplicationInfo appInfo("Hello Triangle", VK_MAKE_VERSION(1, 0, 0),
                                "No Engine", VK_MAKE_VERSION(1, 0, 0),
                                VK_API_VERSION_1_0);

    auto extensions = getRequiredExtensions();
    vk::InstanceCreateInfo createInfo({}, &appInfo, 0, nullptr,
                                      static_cast<uint32_t>(extensions.size()),
                                      extensions.data());

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
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

    for (auto layerName : validationLayers) {
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
    std::vector<vk::PhysicalDevice> devices =
        instance.enumeratePhysicalDevices();

    if (devices.empty()) {
      throw std::runtime_error("No physical devices found");
    }

    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        return;
      }
    }

    throw std::runtime_error("No suitable device found!\n");
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};
    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1);
      queueCreateInfo.pQueuePriorities = &queuePriority;

      queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures{};

    vk::DeviceCreateInfo createInfo;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());

    createInfo.pEnabledFeatures = &deviceFeatures;

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    createInfo.enabledExtensionCount = deviceExtensions.size();
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    device = physicalDevice.createDevice(createInfo);

    graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
  }

  [[nodiscard]] bool isDeviceSuitable(vk::PhysicalDevice physicalDevice) const {
    const auto indices = findQueueFamilies(physicalDevice);

    const bool extensionsSupported =
        checkDeviceExtensionSupport(physicalDevice);

    const auto swapChainSupport = querySwapChainSupport(physicalDevice);

    return indices.isComplete() && swapChainSupport.isSuitable() &&
           extensionsSupported;
  }

  static bool checkDeviceExtensionSupport(vk::PhysicalDevice physicalDevice) {
    std::vector<vk::ExtensionProperties> availableExtensions =
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
      vk::PhysicalDevice device) const {
    QueueFamilyIndices indices;

    std::vector<vk::QueueFamilyProperties> queueFamilies =
        device.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
      auto &queueFamily = queueFamilies[i];

      if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
        indices.graphicsFamily = i;
      }

      if (device.getSurfaceSupportKHR(i, surface) == vk::True) {
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
      vk::PhysicalDevice physicalDevice) const {
    SwapChainSupportDetails details;

    details.capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    details.formats = physicalDevice.getSurfaceFormatsKHR(surface);
    details.presentModes = physicalDevice.getSurfacePresentModesKHR(surface);

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

    auto fileSize = file.tellg();
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
}
