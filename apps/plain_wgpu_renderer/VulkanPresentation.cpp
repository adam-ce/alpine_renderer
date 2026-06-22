/*****************************************************************************
 * Alpine Terrain Renderer
 * Copyright (C) 2026 Adam Celerek <family name at cg tuwien ac at>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include <vulkan/vulkan.h>

#include <dawn/native/VulkanBackend.h>

#include <QDebug>
#include <QVulkanInstance>
#include <QWindow>
#include <QtGlobal>

#include "VulkanPresentation.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <optional>
#include <poll.h>
#include <set>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

void check_vk(VkResult result, const char* what)
{
    if (result != VK_SUCCESS)
        qFatal("%s failed with VkResult %d", what, int(result));
}

void close_fd(int& fd)
{
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

#ifdef Q_OS_ANDROID
void wait_sync_fd(int& fd, const char* what)
{
    if (fd < 0)
        return;

    pollfd sync_poll {};
    sync_poll.fd = fd;
    sync_poll.events = POLLIN;

    int result = 0;
    do {
        result = ::poll(&sync_poll, 1, -1);
    } while (result < 0 && errno == EINTR);

    if (result < 0) {
        const int error = errno;
        close_fd(fd);
        qFatal("%s failed while polling sync FD, errno %d", what, error);
    }

    close_fd(fd);
}
#endif

uint32_t clamp_u32(uint32_t value, uint32_t min, uint32_t max)
{
    return std::max(min, std::min(value, max));
}

WGPUTextureFormat wgpu_format(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_B8G8R8A8_UNORM:
        return WGPUTextureFormat_BGRA8Unorm;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return WGPUTextureFormat_BGRA8UnormSrgb;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return WGPUTextureFormat_RGBA8Unorm;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return WGPUTextureFormat_RGBA8UnormSrgb;
    default:
        qFatal("Unsupported Vulkan swapchain format for Dawn interop: %d", int(format));
    }
}

} // namespace

struct VulkanPresentation::Impl {
    explicit Impl(QVulkanInstance& vulkan_instance, QWindow& window)
        : qt_instance(vulkan_instance)
        , window(window)
    {
    }

    struct QueueFamilies {
        uint32_t graphics = std::numeric_limits<uint32_t>::max();
        uint32_t present = std::numeric_limits<uint32_t>::max();

        [[nodiscard]] bool complete() const
        {
            return graphics != std::numeric_limits<uint32_t>::max() && present != std::numeric_limits<uint32_t>::max();
        }
    };

    struct SharedTarget {
        VkExternalMemoryImageCreateInfo external_image_info { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
        VkImageCreateInfo image_create_info { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize allocation_size = 0;
        uint32_t memory_type_index = 0;
        WGPUSharedTextureMemory shared_memory = nullptr;
        WGPUTexture texture = nullptr;
        WGPUTextureView view = nullptr;
        int dawn_wait_fd = -1;
        VkImageLayout dawn_wait_old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout dawn_wait_new_layout = VK_IMAGE_LAYOUT_GENERAL;
        bool dawn_initialized = false;
        bool dedicated_allocation = true;
    };

    QVulkanInstance& qt_instance;
    QWindow& window;
    WGPUAdapter wgpu_adapter = nullptr;
    WGPUDevice wgpu_device = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    QueueFamilies queues;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchain_format = VK_FORMAT_UNDEFINED;
    VkFormat shared_format = VK_FORMAT_UNDEFINED;
    WGPUTextureFormat dawn_texture_format = WGPUTextureFormat_Undefined;
    VkExtent2D extent {};
    std::vector<VkImage> swapchain_images;
    std::vector<VkImageLayout> swapchain_image_layouts;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkSemaphore image_available = VK_NULL_HANDLE;
    VkSemaphore render_finished = VK_NULL_HANDLE;
    VkFence frame_fence = VK_NULL_HANDLE;
    std::vector<VkSemaphore> retired_semaphores;

    PFN_vkGetMemoryFdKHR vk_get_memory_fd = nullptr;
    PFN_vkGetSemaphoreFdKHR vk_get_semaphore_fd = nullptr;
    PFN_vkImportSemaphoreFdKHR vk_import_semaphore_fd = nullptr;

    SharedTarget shared;

    void initialize(WGPUAdapter adapter, WGPUDevice webgpu_device, const glm::uvec2& requested_size)
    {
#if !(defined(Q_OS_LINUX) || defined(Q_OS_ANDROID))
        Q_UNUSED(adapter);
        Q_UNUSED(webgpu_device);
        Q_UNUSED(requested_size);
        qFatal("plain_wgpu_renderer Vulkan interop is implemented only for Linux and Android");
#else
        wgpu_adapter = adapter;
        wgpu_device = webgpu_device;
        instance = qt_instance.vkInstance();
        if (instance == VK_NULL_HANDLE)
            qFatal("Qt Vulkan instance is not available");

        surface = QVulkanInstance::surfaceForWindow(&window);
        if (surface == VK_NULL_HANDLE)
            qFatal("Qt did not create a Vulkan surface for the window");

        pick_physical_device();
        create_device();
        load_device_functions();
        create_command_pool();
        create_sync_objects();
        recreate_swapchain(requested_size);
        recreate_shared_target();
        release_shared_target_to_dawn(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, false);
#endif
    }

    void destroy()
    {
        if (device != VK_NULL_HANDLE)
            vkDeviceWaitIdle(device);

        destroy_shared_target();
        destroy_swapchain();

        for (VkSemaphore semaphore : retired_semaphores)
            vkDestroySemaphore(device, semaphore, nullptr);
        retired_semaphores.clear();

        if (render_finished != VK_NULL_HANDLE)
            vkDestroySemaphore(device, render_finished, nullptr);
        if (image_available != VK_NULL_HANDLE)
            vkDestroySemaphore(device, image_available, nullptr);
        if (frame_fence != VK_NULL_HANDLE)
            vkDestroyFence(device, frame_fence, nullptr);
        if (command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, command_pool, nullptr);
        if (device != VK_NULL_HANDLE)
            vkDestroyDevice(device, nullptr);

        render_finished = VK_NULL_HANDLE;
        image_available = VK_NULL_HANDLE;
        frame_fence = VK_NULL_HANDLE;
        command_pool = VK_NULL_HANDLE;
        device = VK_NULL_HANDLE;
        physical_device = VK_NULL_HANDLE;
        surface = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
        wgpu_adapter = nullptr;
        wgpu_device = nullptr;
    }

    void resize(const glm::uvec2& requested_size)
    {
        if (device == VK_NULL_HANDLE)
            return;

        vkDeviceWaitIdle(device);
        destroy_shared_target();
        destroy_swapchain();
        recreate_swapchain(requested_size);
        recreate_shared_target();
        release_shared_target_to_dawn(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, false);
    }

    glm::uvec2 current_pixel_size() const
    {
        const auto ratio = window.devicePixelRatio();
        return {
            std::max(1, int(std::lround(window.width() * ratio))),
            std::max(1, int(std::lround(window.height() * ratio))),
        };
    }

    QueueFamilies find_queue_families(VkPhysicalDevice candidate) const
    {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> properties(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_family_count, properties.data());

        QueueFamilies found;
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if ((properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
                found.graphics = i;

            VkBool32 present_supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface, &present_supported);
            if (present_supported)
                found.present = i;

            if (found.complete() && found.graphics == found.present)
                return found;
        }
        return found;
    }

    bool supports_extensions(VkPhysicalDevice candidate, const std::vector<const char*>& required) const
    {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, extensions.data());

        for (const char* required_extension : required) {
            const bool found = std::any_of(extensions.begin(), extensions.end(), [required_extension](const VkExtensionProperties& extension) {
                return std::strcmp(extension.extensionName, required_extension) == 0;
            });
            if (!found)
                return false;
        }
        return true;
    }

    std::vector<const char*> required_device_extensions() const
    {
        std::vector<const char*> extensions {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        };
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        return extensions;
    }

    bool supports_surface(VkPhysicalDevice candidate) const
    {
        uint32_t format_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface, &format_count, nullptr);
        uint32_t present_mode_count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface, &present_mode_count, nullptr);
        return format_count > 0 && present_mode_count > 0;
    }

    void pick_physical_device()
    {
        WGPUAdapterInfo adapter_info {};
        if (wgpuAdapterGetInfo(wgpu_adapter, &adapter_info) != WGPUStatus_Success)
            qFatal("Could not query Dawn adapter info");

        uint32_t device_count = 0;
        check_vk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr), "vkEnumeratePhysicalDevices");
        if (device_count == 0)
            qFatal("No Vulkan physical devices are available");

        std::vector<VkPhysicalDevice> devices(device_count);
        check_vk(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()), "vkEnumeratePhysicalDevices");

        const std::vector<const char*> required_extensions = required_device_extensions();

        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties properties {};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if (properties.vendorID != adapter_info.vendorID || properties.deviceID != adapter_info.deviceID)
                continue;
            QueueFamilies candidate_queues = find_queue_families(candidate);
            if (!candidate_queues.complete())
                continue;
            if (!supports_extensions(candidate, required_extensions))
                continue;
            if (!supports_surface(candidate))
                continue;

            physical_device = candidate;
            queues = candidate_queues;
            qInfo() << "Matched Dawn Vulkan adapter to VkPhysicalDevice" << Qt::hex << properties.vendorID << properties.deviceID << Qt::dec;
            wgpuAdapterInfoFreeMembers(adapter_info);
            return;
        }

        wgpuAdapterInfoFreeMembers(adapter_info);
        qFatal("No VkPhysicalDevice matches Dawn's Vulkan adapter and Qt's surface requirements");
    }

    void create_device()
    {
        std::vector<VkDeviceQueueCreateInfo> queue_infos;
        std::set<uint32_t> unique_families { queues.graphics, queues.present };
        float priority = 1.0f;
        for (uint32_t family : unique_families) {
            VkDeviceQueueCreateInfo queue_info { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
            queue_info.queueFamilyIndex = family;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &priority;
            queue_infos.push_back(queue_info);
        }

        const std::vector<const char*> extensions = required_device_extensions();

        VkDeviceCreateInfo create_info { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        create_info.queueCreateInfoCount = uint32_t(queue_infos.size());
        create_info.pQueueCreateInfos = queue_infos.data();
        create_info.enabledExtensionCount = uint32_t(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        check_vk(vkCreateDevice(physical_device, &create_info, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queues.graphics, 0, &graphics_queue);
        vkGetDeviceQueue(device, queues.present, 0, &present_queue);
    }

    void load_device_functions()
    {
        vk_get_memory_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        vk_get_semaphore_fd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
        vk_import_semaphore_fd = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(vkGetDeviceProcAddr(device, "vkImportSemaphoreFdKHR"));
        if (!vk_get_memory_fd || !vk_get_semaphore_fd || !vk_import_semaphore_fd)
            qFatal("Vulkan external FD entry points are not available");
    }

    void create_command_pool()
    {
        VkCommandPoolCreateInfo pool_info { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = queues.graphics;
        check_vk(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo allocate_info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate_info.commandPool = command_pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;
        check_vk(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer), "vkAllocateCommandBuffers");
    }

    VkSemaphore create_semaphore(bool exportable)
    {
        VkExportSemaphoreCreateInfo export_info { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
        export_info.handleTypes = semaphore_handle_type();

        VkSemaphoreCreateInfo semaphore_info { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        if (exportable)
            semaphore_info.pNext = &export_info;

        VkSemaphore semaphore = VK_NULL_HANDLE;
        check_vk(vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore), "vkCreateSemaphore");
        return semaphore;
    }

    VkExternalSemaphoreHandleTypeFlagBits semaphore_handle_type() const
    {
#ifdef Q_OS_ANDROID
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
#else
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    }

    void create_sync_objects()
    {
        image_available = create_semaphore(false);
        render_finished = create_semaphore(false);

        VkFenceCreateInfo fence_info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        check_vk(vkCreateFence(device, &fence_info, nullptr, &frame_fence), "vkCreateFence");
    }

    VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) const
    {
        const std::array<VkFormat, 4> preferred {
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_B8G8R8A8_SRGB,
            VK_FORMAT_R8G8B8A8_SRGB,
        };
        for (VkFormat preferred_format : preferred) {
            auto it = std::find_if(formats.begin(), formats.end(), [preferred_format](const VkSurfaceFormatKHR& format) {
                return format.format == preferred_format && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            });
            if (it != formats.end())
                return *it;
        }
        for (const VkSurfaceFormatKHR& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM || format.format == VK_FORMAT_R8G8B8A8_UNORM
                || format.format == VK_FORMAT_B8G8R8A8_SRGB || format.format == VK_FORMAT_R8G8B8A8_SRGB)
                return format;
        }
        qFatal("Qt Vulkan surface does not expose an RGBA/BGRA 8-bit swapchain format");
    }

    VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& capabilities, const glm::uvec2& requested_size) const
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        return {
            clamp_u32(requested_size.x, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            clamp_u32(requested_size.y, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        };
    }

    void recreate_swapchain(const glm::uvec2& requested_size)
    {
        VkSurfaceCapabilitiesKHR capabilities {};
        check_vk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0)
            qFatal("Qt Vulkan swapchain does not support transfer-dst presentation");

        uint32_t format_count = 0;
        check_vk(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr), "vkGetPhysicalDeviceSurfaceFormatsKHR");
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        check_vk(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats.data()), "vkGetPhysicalDeviceSurfaceFormatsKHR");

        VkSurfaceFormatKHR chosen_format = choose_surface_format(formats);
        swapchain_format = chosen_format.format;
        extent = choose_extent(capabilities, requested_size);

        uint32_t image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0)
            image_count = std::min(image_count, capabilities.maxImageCount);

        uint32_t queue_family_indices[] = { queues.graphics, queues.present };
        VkSwapchainCreateInfoKHR create_info { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        create_info.surface = surface;
        create_info.minImageCount = image_count;
        create_info.imageFormat = swapchain_format;
        create_info.imageColorSpace = chosen_format.colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (queues.graphics != queues.present) {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queue_family_indices;
        } else {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        create_info.preTransform = capabilities.currentTransform;
        create_info.compositeAlpha = (capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0
            ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
            : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        create_info.clipped = VK_TRUE;

        check_vk(vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain), "vkCreateSwapchainKHR");

        uint32_t swapchain_image_count = 0;
        check_vk(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr), "vkGetSwapchainImagesKHR");
        swapchain_images.resize(swapchain_image_count);
        check_vk(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images.data()), "vkGetSwapchainImagesKHR");
        swapchain_image_layouts.assign(swapchain_images.size(), VK_IMAGE_LAYOUT_UNDEFINED);
        choose_shared_format();
    }

    void destroy_swapchain()
    {
        swapchain_images.clear();
        swapchain_image_layouts.clear();
        if (swapchain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
        swapchain_format = VK_FORMAT_UNDEFINED;
        shared_format = VK_FORMAT_UNDEFINED;
        dawn_texture_format = WGPUTextureFormat_Undefined;
        extent = {};
    }

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties memory_properties {};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if ((type_filter & (1u << i)) != 0 && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        qFatal("Could not find a suitable Vulkan memory type");
    }

    bool external_image_support(VkFormat format, bool* dedicated_only = nullptr) const
    {
        VkPhysicalDeviceExternalImageFormatInfo external_info { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO };
        external_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkPhysicalDeviceImageFormatInfo2 format_info { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2 };
        format_info.pNext = &external_info;
        format_info.format = format;
        format_info.type = VK_IMAGE_TYPE_2D;
        format_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        format_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        format_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

        VkExternalImageFormatProperties external_properties { VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES };
        VkImageFormatProperties2 properties { VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
        properties.pNext = &external_properties;
        VkResult result = vkGetPhysicalDeviceImageFormatProperties2(physical_device, &format_info, &properties);
        if (result == VK_ERROR_FORMAT_NOT_SUPPORTED)
            return false;
        check_vk(result, "vkGetPhysicalDeviceImageFormatProperties2");
        if ((external_properties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0)
            return false;
        if (dedicated_only)
            *dedicated_only = (external_properties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
        return true;
    }

    bool has_blit_support(VkFormat format, VkFormatFeatureFlags feature) const
    {
        VkFormatProperties properties {};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        return (properties.optimalTilingFeatures & feature) != 0;
    }

    void choose_shared_format()
    {
        std::vector<VkFormat> candidates {
            swapchain_format,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_FORMAT_B8G8R8A8_SRGB,
        };
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        for (VkFormat candidate : candidates) {
            bool dedicated_only = false;
            if (!external_image_support(candidate, &dedicated_only))
                continue;
            if (candidate != swapchain_format) {
                if (!has_blit_support(candidate, VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !has_blit_support(swapchain_format, VK_FORMAT_FEATURE_BLIT_DST_BIT))
                    continue;
            }
            shared_format = candidate;
            dawn_texture_format = wgpu_format(shared_format);
            qInfo() << "Using shared image format" << shared_format << "for swapchain format" << swapchain_format
                    << "dedicated-only:" << dedicated_only;
            return;
        }

        qFatal("No exportable Vulkan color format is available for Dawn interop");
    }

    void recreate_shared_target()
    {
        shared.external_image_info = VkExternalMemoryImageCreateInfo { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
        shared.external_image_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        shared.image_create_info = VkImageCreateInfo { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        shared.image_create_info.pNext = &shared.external_image_info;
        shared.image_create_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        shared.image_create_info.imageType = VK_IMAGE_TYPE_2D;
        shared.image_create_info.format = shared_format;
        shared.image_create_info.extent = { extent.width, extent.height, 1 };
        shared.image_create_info.mipLevels = 1;
        shared.image_create_info.arrayLayers = 1;
        shared.image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
        shared.image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        shared.image_create_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        shared.image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        shared.image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        check_vk(vkCreateImage(device, &shared.image_create_info, nullptr, &shared.image), "vkCreateImage(shared)");

        VkMemoryRequirements requirements {};
        vkGetImageMemoryRequirements(device, shared.image, &requirements);
        shared.allocation_size = requirements.size;
        shared.memory_type_index = find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkMemoryDedicatedAllocateInfo dedicated_info { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
        dedicated_info.image = shared.image;

        VkExportMemoryAllocateInfo export_info { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
        export_info.pNext = &dedicated_info;
        export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo allocate_info { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocate_info.pNext = &export_info;
        allocate_info.allocationSize = shared.allocation_size;
        allocate_info.memoryTypeIndex = shared.memory_type_index;

        check_vk(vkAllocateMemory(device, &allocate_info, nullptr, &shared.memory), "vkAllocateMemory(shared)");
        check_vk(vkBindImageMemory(device, shared.image, shared.memory, 0), "vkBindImageMemory(shared)");
        shared.dedicated_allocation = true;
    }

    int export_memory_fd(VkDeviceMemory memory)
    {
        VkMemoryGetFdInfoKHR fd_info { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
        fd_info.memory = memory;
        fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        check_vk(vk_get_memory_fd(device, &fd_info, &fd), "vkGetMemoryFdKHR");
        return fd;
    }

    WGPUTextureDescriptor shared_texture_descriptor() const
    {
        WGPUTextureDescriptor texture_desc {};
        texture_desc.label = WGPUStringView { .data = "plain renderer Dawn render target", .length = WGPU_STRLEN };
        texture_desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
        texture_desc.dimension = WGPUTextureDimension_2D;
        texture_desc.size = { extent.width, extent.height, 1 };
        texture_desc.format = dawn_texture_format;
        texture_desc.mipLevelCount = 1;
        texture_desc.sampleCount = 1;
        return texture_desc;
    }

    void create_shared_texture_view()
    {
        WGPUTextureViewDescriptor view_desc {};
        view_desc.label = WGPUStringView { .data = "plain renderer Dawn render target view", .length = WGPU_STRLEN };
        view_desc.format = dawn_texture_format;
        view_desc.dimension = WGPUTextureViewDimension_2D;
        view_desc.mipLevelCount = 1;
        view_desc.arrayLayerCount = 1;
        view_desc.aspect = WGPUTextureAspect_All;
        shared.view = wgpuTextureCreateView(shared.texture, &view_desc);
        if (!shared.view)
            qFatal("wgpuTextureCreateView(shared render target) failed");
    }

#ifdef Q_OS_ANDROID
    void import_android_shared_texture_memory()
    {
        if (shared.shared_memory)
            return;

        WGPUSharedTextureMemoryOpaqueFDDescriptor fd_desc {};
        fd_desc.chain.sType = WGPUSType_SharedTextureMemoryOpaqueFDDescriptor;
        fd_desc.vkImageCreateInfo = &shared.image_create_info;
        fd_desc.memoryFD = export_memory_fd(shared.memory);
        fd_desc.memoryTypeIndex = shared.memory_type_index;
        fd_desc.allocationSize = shared.allocation_size;
        fd_desc.dedicatedAllocation = shared.dedicated_allocation;

        WGPUSharedTextureMemoryDescriptor descriptor {};
        descriptor.nextInChain = &fd_desc.chain;
        descriptor.label = WGPUStringView { .data = "plain renderer Android shared texture memory", .length = WGPU_STRLEN };

        shared.shared_memory = wgpuDeviceImportSharedTextureMemory(wgpu_device, &descriptor);
        if (!shared.shared_memory) {
            close_fd(fd_desc.memoryFD);
            qFatal("wgpuDeviceImportSharedTextureMemory(opaque FD) failed");
        }
    }

    WGPUSharedFence import_android_wait_fence()
    {
        if (shared.dawn_wait_fd < 0)
            return nullptr;

        WGPUSharedFenceSyncFDDescriptor sync_desc {};
        sync_desc.chain.sType = WGPUSType_SharedFenceSyncFDDescriptor;
        sync_desc.handle = shared.dawn_wait_fd;

        WGPUSharedFenceDescriptor descriptor {};
        descriptor.nextInChain = &sync_desc.chain;

        WGPUSharedFence fence = wgpuDeviceImportSharedFence(wgpu_device, &descriptor);
        if (!fence) {
            close_fd(shared.dawn_wait_fd);
            qFatal("wgpuDeviceImportSharedFence(sync FD) failed");
        }

        shared.dawn_wait_fd = -1;
        return fence;
    }
#endif

    void destroy_shared_target()
    {
        close_fd(shared.dawn_wait_fd);
        if (shared.view)
            wgpuTextureViewRelease(shared.view);
        if (shared.texture)
            wgpuTextureRelease(shared.texture);
        if (shared.shared_memory)
            wgpuSharedTextureMemoryRelease(shared.shared_memory);
        if (shared.memory != VK_NULL_HANDLE)
            vkFreeMemory(device, shared.memory, nullptr);
        if (shared.image != VK_NULL_HANDLE)
            vkDestroyImage(device, shared.image, nullptr);
        shared = SharedTarget {};
    }

    VkSemaphore import_semaphore_fd(int& fd)
    {
        VkSemaphore semaphore = create_semaphore(false);
        VkImportSemaphoreFdInfoKHR import_info { VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR };
        import_info.semaphore = semaphore;
        import_info.flags = semaphore_handle_type() == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT ? VK_SEMAPHORE_IMPORT_TEMPORARY_BIT : 0;
        import_info.handleType = semaphore_handle_type();
        import_info.fd = fd;
        VkResult result = vk_import_semaphore_fd(device, &import_info);
        if (result != VK_SUCCESS) {
            vkDestroySemaphore(device, semaphore, nullptr);
            close_fd(fd);
            qFatal("vkImportSemaphoreFdKHR failed with VkResult %d", int(result));
        }
        fd = -1;
        return semaphore;
    }

    int export_semaphore_fd(VkSemaphore semaphore)
    {
        VkSemaphoreGetFdInfoKHR fd_info { VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
        fd_info.semaphore = semaphore;
        fd_info.handleType = semaphore_handle_type();
        int fd = -1;
        check_vk(vk_get_semaphore_fd(device, &fd_info, &fd), "vkGetSemaphoreFdKHR");
        return fd;
    }

    void begin_single_time_commands()
    {
        check_vk(vkResetCommandBuffer(command_buffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin_info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
    }

    void release_shared_target_to_dawn(VkImageLayout old_layout, VkImageLayout new_layout, bool initialized)
    {
        begin_single_time_commands();
        VkImageMemoryBarrier release_barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        release_barrier.srcAccessMask = old_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ? VK_ACCESS_TRANSFER_READ_BIT : 0;
        release_barrier.dstAccessMask = 0;
        release_barrier.oldLayout = old_layout;
        release_barrier.newLayout = new_layout;
        release_barrier.srcQueueFamilyIndex = queues.graphics;
        release_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL_KHR;
        release_barrier.image = shared.image;
        release_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        release_barrier.subresourceRange.baseMipLevel = 0;
        release_barrier.subresourceRange.levelCount = 1;
        release_barrier.subresourceRange.baseArrayLayer = 0;
        release_barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = old_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        vkCmdPipelineBarrier(command_buffer, src_stage, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release_barrier);
        check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        VkSemaphore signal = create_semaphore(true);
        VkSubmitInfo submit_info { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &signal;

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fence_info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        check_vk(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence(initial shared release)");
        check_vk(vkQueueSubmit(graphics_queue, 1, &submit_info, fence), "vkQueueSubmit(initial shared release)");
        check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()), "vkWaitForFences(initial shared release)");
        vkDestroyFence(device, fence, nullptr);

        shared.dawn_wait_fd = export_semaphore_fd(signal);
        vkDestroySemaphore(device, signal, nullptr);
        shared.dawn_wait_old_layout = old_layout;
        shared.dawn_wait_new_layout = new_layout;
        shared.dawn_initialized = initialized;
    }

    void begin_dawn_access()
    {
        if (shared.texture || shared.view)
            qFatal("Dawn access is already open for the shared Vulkan image");

        WGPUTextureDescriptor texture_desc = shared_texture_descriptor();
#ifdef Q_OS_ANDROID
        import_android_shared_texture_memory();

        shared.texture = wgpuSharedTextureMemoryCreateTexture(shared.shared_memory, &texture_desc);
        if (!shared.texture)
            qFatal("wgpuSharedTextureMemoryCreateTexture failed");

        WGPUSharedFence wait_fence = import_android_wait_fence();
        uint64_t wait_value = 1;

        WGPUSharedTextureMemoryVkImageLayoutBeginState layout_begin {};
        layout_begin.chain.sType = WGPUSType_SharedTextureMemoryVkImageLayoutBeginState;
        layout_begin.oldLayout = int32_t(shared.dawn_wait_old_layout);
        layout_begin.newLayout = int32_t(shared.dawn_wait_new_layout);

        WGPUSharedTextureMemoryBeginAccessDescriptor begin {};
        begin.nextInChain = &layout_begin.chain;
        begin.initialized = shared.dawn_initialized;
        if (wait_fence) {
            begin.fenceCount = 1;
            begin.fences = &wait_fence;
            begin.signaledValues = &wait_value;
        }

        WGPUStatus begin_status = wgpuSharedTextureMemoryBeginAccess(shared.shared_memory, shared.texture, &begin);
        if (wait_fence)
            wgpuSharedFenceRelease(wait_fence);
        if (begin_status != WGPUStatus_Success)
            qFatal("wgpuSharedTextureMemoryBeginAccess failed with status %d", int(begin_status));
#else
        dawn::native::vulkan::ExternalImageDescriptorOpaqueFD descriptor;
        descriptor.cTextureDescriptor = &texture_desc;
        descriptor.isInitialized = shared.dawn_initialized;
        descriptor.memoryFD = export_memory_fd(shared.memory);
        descriptor.allocationSize = shared.allocation_size;
        descriptor.memoryTypeIndex = shared.memory_type_index;
        descriptor.releasedOldLayout = shared.dawn_wait_old_layout;
        descriptor.releasedNewLayout = shared.dawn_wait_new_layout;
        descriptor.dedicatedAllocation = shared.dedicated_allocation
            ? dawn::native::vulkan::NeedsDedicatedAllocation::Yes
            : dawn::native::vulkan::NeedsDedicatedAllocation::No;
        if (shared.dawn_wait_fd >= 0)
            descriptor.waitFDs.push_back(shared.dawn_wait_fd);

        shared.texture = dawn::native::vulkan::WrapVulkanImage(wgpu_device, &descriptor);
        if (!shared.texture) {
            close_fd(descriptor.memoryFD);
            for (int& fd : descriptor.waitFDs)
                close_fd(fd);
            shared.dawn_wait_fd = -1;
            qFatal("dawn::native::vulkan::WrapVulkanImage failed");
        }
        shared.dawn_wait_fd = -1;
#endif
        create_shared_texture_view();
    }

    void end_dawn_access_and_present()
    {
        if (!shared.texture || !shared.view)
            qFatal("Dawn access is not open for the shared Vulkan image");

        wgpuTextureViewRelease(shared.view);
        shared.view = nullptr;

#ifdef Q_OS_ANDROID
        WGPUSharedTextureMemoryVkImageLayoutEndState layout_end {};
        layout_end.chain.sType = WGPUSType_SharedTextureMemoryVkImageLayoutEndState;

        WGPUSharedTextureMemoryEndAccessState end {};
        end.nextInChain = &layout_end.chain;
        WGPUStatus end_status = wgpuSharedTextureMemoryEndAccess(shared.shared_memory, shared.texture, &end);
        wgpuTextureRelease(shared.texture);
        shared.texture = nullptr;

        if (end_status != WGPUStatus_Success)
            qFatal("wgpuSharedTextureMemoryEndAccess failed with status %d", int(end_status));
        if (end.fenceCount != 1)
            qFatal("Expected one Dawn release fence, got %zu", end.fenceCount);

        WGPUSharedFenceSyncFDExportInfo sync_export {};
        sync_export.chain.sType = WGPUSType_SharedFenceSyncFDExportInfo;
        sync_export.handle = -1;

        WGPUSharedFenceExportInfo fence_export {};
        fence_export.nextInChain = &sync_export.chain;
        wgpuSharedFenceExportInfo(end.fences[0], &fence_export);

        int dawn_release_fd = sync_export.handle;
        VkImageLayout released_old_layout = VkImageLayout(layout_end.oldLayout);
        VkImageLayout released_new_layout = VkImageLayout(layout_end.newLayout);

        wgpuSharedTextureMemoryEndAccessStateFreeMembers(end);

        if (dawn_release_fd < 0)
            qFatal("Dawn did not export a sync FD release fence");
#else
        dawn::native::vulkan::ExternalImageExportInfoOpaqueFD export_info;
        if (!dawn::native::vulkan::ExportVulkanImage(shared.texture, &export_info))
            qFatal("dawn::native::vulkan::ExportVulkanImage failed");
        wgpuTextureRelease(shared.texture);
        shared.texture = nullptr;

        if (export_info.semaphoreHandles.size() != 1)
            qFatal("Expected one Dawn release semaphore, got %zu", export_info.semaphoreHandles.size());

        int dawn_release_fd = export_info.semaphoreHandles[0];
        VkImageLayout released_old_layout = export_info.releasedOldLayout;
        VkImageLayout released_new_layout = export_info.releasedNewLayout;
#endif

        present_after_dawn(dawn_release_fd, released_old_layout, released_new_layout);
    }

    void cleanup_retired_semaphores()
    {
        for (VkSemaphore semaphore : retired_semaphores)
            vkDestroySemaphore(device, semaphore, nullptr);
        retired_semaphores.clear();
    }

    void record_copy_to_swapchain(uint32_t image_index, VkImageLayout dawn_old_layout, VkImageLayout dawn_new_layout)
    {
        begin_single_time_commands();

        VkImageMemoryBarrier acquire_barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        acquire_barrier.srcAccessMask = 0;
        acquire_barrier.dstAccessMask = 0;
        acquire_barrier.oldLayout = dawn_old_layout;
        acquire_barrier.newLayout = dawn_new_layout;
        acquire_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL_KHR;
        acquire_barrier.dstQueueFamilyIndex = queues.graphics;
        acquire_barrier.image = shared.image;
        acquire_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        acquire_barrier.subresourceRange.baseMipLevel = 0;
        acquire_barrier.subresourceRange.levelCount = 1;
        acquire_barrier.subresourceRange.baseArrayLayer = 0;
        acquire_barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquire_barrier);

        if (dawn_new_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            VkImageMemoryBarrier transfer_src_barrier = acquire_barrier;
            transfer_src_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            transfer_src_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            transfer_src_barrier.oldLayout = dawn_new_layout;
            transfer_src_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            transfer_src_barrier.srcQueueFamilyIndex = queues.graphics;
            transfer_src_barrier.dstQueueFamilyIndex = queues.graphics;
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &transfer_src_barrier);
        }

        VkImageMemoryBarrier swapchain_to_dst { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        swapchain_to_dst.srcAccessMask = 0;
        swapchain_to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapchain_to_dst.oldLayout = swapchain_image_layouts[image_index];
        swapchain_to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapchain_to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapchain_to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapchain_to_dst.image = swapchain_images[image_index];
        swapchain_to_dst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        swapchain_to_dst.subresourceRange.baseMipLevel = 0;
        swapchain_to_dst.subresourceRange.levelCount = 1;
        swapchain_to_dst.subresourceRange.baseArrayLayer = 0;
        swapchain_to_dst.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapchain_to_dst);

        if (shared_format == swapchain_format) {
            VkImageCopy copy_region {};
            copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy_region.srcSubresource.layerCount = 1;
            copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy_region.dstSubresource.layerCount = 1;
            copy_region.extent = { extent.width, extent.height, 1 };
            vkCmdCopyImage(command_buffer,
                shared.image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                swapchain_images[image_index],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &copy_region);
        } else {
            VkImageBlit blit_region {};
            blit_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit_region.srcSubresource.layerCount = 1;
            blit_region.srcOffsets[1] = { int32_t(extent.width), int32_t(extent.height), 1 };
            blit_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit_region.dstSubresource.layerCount = 1;
            blit_region.dstOffsets[1] = { int32_t(extent.width), int32_t(extent.height), 1 };
            vkCmdBlitImage(command_buffer,
                shared.image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                swapchain_images[image_index],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit_region,
                VK_FILTER_NEAREST);
        }

        VkImageMemoryBarrier swapchain_to_present = swapchain_to_dst;
        swapchain_to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapchain_to_present.dstAccessMask = 0;
        swapchain_to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapchain_to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapchain_to_present);

#ifndef Q_OS_ANDROID
        VkImageMemoryBarrier release_barrier = acquire_barrier;
        release_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        release_barrier.dstAccessMask = 0;
        release_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        release_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        release_barrier.srcQueueFamilyIndex = queues.graphics;
        release_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL_KHR;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release_barrier);
#endif

        check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
        swapchain_image_layouts[image_index] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    void present_after_dawn(int dawn_release_fd, VkImageLayout dawn_old_layout, VkImageLayout dawn_new_layout)
    {
        if (dawn_release_fd < 0)
            qFatal("Dawn did not export a Vulkan semaphore FD");

        check_vk(vkWaitForFences(device, 1, &frame_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()), "vkWaitForFences(frame)");
        cleanup_retired_semaphores();
        check_vk(vkResetFences(device, 1, &frame_fence), "vkResetFences(frame)");

#ifdef Q_OS_ANDROID
        wait_sync_fd(dawn_release_fd, "Dawn release sync FD wait");
        VkSemaphore dawn_wait = VK_NULL_HANDLE;
#else
        VkSemaphore dawn_wait = import_semaphore_fd(dawn_release_fd);
#endif

        uint32_t image_index = 0;
        VkResult acquire_result = vkAcquireNextImageKHR(device, swapchain, std::numeric_limits<uint64_t>::max(), image_available, VK_NULL_HANDLE, &image_index);
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
            if (dawn_wait != VK_NULL_HANDLE)
                vkDestroySemaphore(device, dawn_wait, nullptr);
            resize(current_pixel_size());
            return;
        }
        if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
            if (dawn_wait != VK_NULL_HANDLE)
                vkDestroySemaphore(device, dawn_wait, nullptr);
            qFatal("vkAcquireNextImageKHR failed with VkResult %d", int(acquire_result));
        }

        record_copy_to_swapchain(image_index, dawn_old_layout, dawn_new_layout);

#ifdef Q_OS_ANDROID
        std::array<VkSemaphore, 1> wait_semaphores { image_available };
        std::array<VkPipelineStageFlags, 1> wait_stages { VK_PIPELINE_STAGE_TRANSFER_BIT };
#else
        std::array<VkSemaphore, 2> wait_semaphores { image_available, dawn_wait };
        std::array<VkPipelineStageFlags, 2> wait_stages { VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT };
#endif
#ifdef Q_OS_ANDROID
        VkSubmitInfo submit_info { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit_info.waitSemaphoreCount = uint32_t(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_finished;
        check_vk(vkQueueSubmit(graphics_queue, 1, &submit_info, frame_fence), "vkQueueSubmit(copy to swapchain)");

        check_vk(vkWaitForFences(device, 1, &frame_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()), "vkWaitForFences(copy to swapchain)");
        release_shared_target_to_dawn(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, true);
#else
        VkSemaphore dawn_release = create_semaphore(true);
        std::array<VkSemaphore, 2> signal_semaphores { render_finished, dawn_release };

        VkSubmitInfo submit_info { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit_info.waitSemaphoreCount = uint32_t(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        submit_info.signalSemaphoreCount = uint32_t(signal_semaphores.size());
        submit_info.pSignalSemaphores = signal_semaphores.data();
        check_vk(vkQueueSubmit(graphics_queue, 1, &submit_info, frame_fence), "vkQueueSubmit(copy to swapchain)");

        shared.dawn_wait_fd = export_semaphore_fd(dawn_release);
        shared.dawn_wait_old_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        shared.dawn_wait_new_layout = VK_IMAGE_LAYOUT_GENERAL;
        shared.dawn_initialized = true;
#endif

        VkPresentInfoKHR present_info { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain;
        present_info.pImageIndices = &image_index;
        VkResult present_result = vkQueuePresentKHR(present_queue, &present_info);
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
            qWarning() << "Vulkan swapchain became outdated during present";
        } else if (present_result != VK_SUCCESS) {
            qFatal("vkQueuePresentKHR failed with VkResult %d", int(present_result));
        }

        if (dawn_wait != VK_NULL_HANDLE)
            retired_semaphores.push_back(dawn_wait);
#ifndef Q_OS_ANDROID
        retired_semaphores.push_back(dawn_release);
#endif
    }
};

VulkanPresentation::VulkanPresentation(QVulkanInstance& vulkan_instance, QWindow& window)
    : m_impl(std::make_unique<Impl>(vulkan_instance, window))
{
}

VulkanPresentation::~VulkanPresentation() { destroy(); }

void VulkanPresentation::initialize(WGPUAdapter adapter, WGPUDevice device, const glm::uvec2& size) { m_impl->initialize(adapter, device, size); }

void VulkanPresentation::resize(const glm::uvec2& size) { m_impl->resize(size); }

void VulkanPresentation::destroy()
{
    if (m_impl)
        m_impl->destroy();
}

WGPUTextureFormat VulkanPresentation::texture_format() const { return m_impl->dawn_texture_format; }

glm::uvec2 VulkanPresentation::size() const { return { m_impl->extent.width, m_impl->extent.height }; }

WGPUTextureView VulkanPresentation::render_target_view() const { return m_impl->shared.view; }

void VulkanPresentation::begin_dawn_access() { m_impl->begin_dawn_access(); }

void VulkanPresentation::end_dawn_access_and_present() { m_impl->end_dawn_access_and_present(); }
