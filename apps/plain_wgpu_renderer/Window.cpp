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

#include "Window.h"

#include "nucleus/tile/drawing.h"
#include "webgpu/base/RenderResourceRegistry.h"
#include "webgpu/base/raii/RenderPassEncoder.h"
#include "webgpu/base/webgpu_interface.hpp"
#include "webgpu/engine/tile_mesh/TileMeshRenderer.h"

#include <QCloseEvent>
#include <QDebug>
#include <QExposeEvent>
#include <QJniEnvironment>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QTouchEvent>
#include <QWheelEvent>
#include <algorithm>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <cmath>

namespace {

void log_jni_exception(JNIEnv* env, const char* context)
{
    if (!env->ExceptionCheck())
        return;
    qWarning() << context;
    env->ExceptionDescribe();
    env->ExceptionClear();
}

void webgpu_device_error_callback(
    [[maybe_unused]] const WGPUDevice* device, WGPUErrorType type, WGPUStringView message, [[maybe_unused]] void* userdata1, [[maybe_unused]] void* userdata2)
{
    qCritical() << "WebGPU error" << type << QByteArray(message.data, qsizetype(message.length));
}

void webgpu_device_lost_callback([[maybe_unused]] const WGPUDevice* device,
    WGPUDeviceLostReason reason,
    WGPUStringView message,
    [[maybe_unused]] void* userdata1,
    [[maybe_unused]] void* userdata2)
{
    qCritical() << "WebGPU device lost" << reason << QByteArray(message.data, qsizetype(message.length));
}

} // namespace

Window::Window(std::shared_ptr<webgpu_engine::Context> context)
    : QWindow()
    , m_context(std::move(context))
{
    setTitle(QStringLiteral("AlpineMaps WebGPU"));
    setSurfaceType(QSurface::VulkanSurface);
    resize(1280, 720);

    if (!m_vulkan_instance.create())
        qFatal("Could not create Qt Vulkan instance");
    setVulkanInstance(&m_vulkan_instance);

    m_render_timer.setSingleShot(true);
    m_render_timer.setInterval(0);
    connect(&m_render_timer, &QTimer::timeout, this, &Window::render);
}

Window::~Window() { destroy(); }

void Window::initialise_gpu()
{
    if (m_initialized || m_destroyed || !isExposed())
        return;

    create_webgpu_context();
    m_context->set_webgpu_ctx(m_webgpu_context);

    m_context->shared_config().m_atmosphere_enabled = false;
    m_context->shared_config().m_clouds_enabled = false;
    m_context->shared_config().m_shading_enabled = false;
    m_context->shared_config().m_overlay_mode = 0;
    m_context->shared_config().m_track_render_mode = 0;

    emit initialisation_started();

    create_buffers();
    configure_surface(pixel_size());
    m_context->webgpu_ctx().resource_registry().recreate_all(m_device);
    create_bind_groups();
    create_compose_pipeline();

    resize_framebuffer(pixel_size());

    m_initialized = true;
    update_camera(m_camera);
    request_render();
}

void Window::destroy()
{
    if (m_destroyed)
        return;
    m_destroyed = true;

    emit about_to_be_destoryed();

    m_gbuffer.reset();
    m_compose_bind_group.reset();
    m_compose_pipeline.reset();
    m_compose_shader.reset();
    m_compose_bind_group_layout.reset();
    m_camera_bind_group.reset();
    m_shared_config_bind_group.reset();
    m_camera_config_ubo.reset();
    m_shared_config_ubo.reset();

    release_webgpu_context();
}

float Window::depth([[maybe_unused]] const glm::dvec2& normalised_device_coordinates) { return 0.0f; }

glm::dvec3 Window::position([[maybe_unused]] const glm::dvec2& normalised_device_coordinates) { return m_camera.position(); }

void Window::update_camera(const nucleus::camera::Definition& new_definition)
{
    m_camera = new_definition;
    if (!m_initialized || !m_camera_config_ubo)
        return;

    webgpu_engine::uboCameraConfig* cc = &m_camera_config_ubo->data;
    cc->position = glm::vec4(new_definition.position(), 1.0);
    cc->view_matrix = new_definition.local_view_matrix();
    cc->proj_matrix = new_definition.projection_matrix();
    cc->view_proj_matrix = cc->proj_matrix * cc->view_matrix;
    cc->inv_view_proj_matrix = glm::inverse(cc->view_proj_matrix);
    cc->inv_view_matrix = glm::inverse(cc->view_matrix);
    cc->inv_proj_matrix = glm::inverse(cc->proj_matrix);
    cc->viewport_size = new_definition.viewport_size();
    cc->distance_scaling_factor = new_definition.distance_scale_factor();
    m_camera_config_ubo->update_gpu_data(m_queue);

    request_render();
}

void Window::request_render()
{
    if (!m_initialized || m_destroyed || m_render_queued)
        return;
    m_render_queued = true;
    m_render_timer.start();
}

void Window::exposeEvent(QExposeEvent* event)
{
    QWindow::exposeEvent(event);
    if (isExposed())
        initialise_gpu();
}

void Window::resizeEvent(QResizeEvent* event)
{
    QWindow::resizeEvent(event);
    emit resized(logical_size());
    if (!m_initialized)
        return;
    configure_surface(pixel_size());
    resize_framebuffer(pixel_size());
    request_render();
}

void Window::closeEvent(QCloseEvent* event)
{
    destroy();
    QWindow::closeEvent(event);
}

void Window::mousePressEvent(QMouseEvent* event) { emit mouse_pressed(nucleus::event_parameter::make(event)); }

void Window::mouseMoveEvent(QMouseEvent* event) { emit mouse_moved(nucleus::event_parameter::make(event)); }

void Window::wheelEvent(QWheelEvent* event) { emit wheel_turned(nucleus::event_parameter::make(event)); }

void Window::keyPressEvent(QKeyEvent* event)
{
    if (!event->isAutoRepeat())
        emit key_pressed(event->keyCombination());
}

void Window::keyReleaseEvent(QKeyEvent* event)
{
    if (!event->isAutoRepeat())
        emit key_released(event->keyCombination());
}

void Window::touchEvent(QTouchEvent* event) { emit touch_made(nucleus::event_parameter::make(event)); }

glm::uvec2 Window::logical_size() const { return { std::max(1, width()), std::max(1, height()) }; }

glm::uvec2 Window::pixel_size() const
{
    const auto ratio = devicePixelRatio();
    return {
        std::max(1, int(std::lround(width() * ratio))),
        std::max(1, int(std::lround(height() * ratio))),
    };
}

ANativeWindow* Window::android_native_window() const
{
    QJniEnvironment jni;
    JNIEnv* env = jni.jniEnv();
    jobject qt_window = reinterpret_cast<jobject>(winId());
    if (!qt_window)
        return nullptr;

    jclass qt_window_class = env->GetObjectClass(qt_window);
    log_jni_exception(env, "Could not get QtWindow class");
    if (!qt_window_class)
        return nullptr;

    jfieldID surface_container_field = env->GetFieldID(qt_window_class, "m_surfaceContainer", "Landroid/view/View;");
    log_jni_exception(env, "Could not get QtWindow.m_surfaceContainer");
    if (!surface_container_field) {
        env->DeleteLocalRef(qt_window_class);
        return nullptr;
    }

    jobject surface_container = env->GetObjectField(qt_window, surface_container_field);
    log_jni_exception(env, "Could not get QtWindow surface container");
    env->DeleteLocalRef(qt_window_class);
    if (!surface_container)
        return nullptr;

    jclass surface_view_class = env->FindClass("android/view/SurfaceView");
    log_jni_exception(env, "Could not find android.view.SurfaceView");
    if (!surface_view_class) {
        env->DeleteLocalRef(surface_container);
        return nullptr;
    }

    if (!env->IsInstanceOf(surface_container, surface_view_class)) {
        qWarning() << "Qt Android surface container is not a SurfaceView";
        env->DeleteLocalRef(surface_view_class);
        env->DeleteLocalRef(surface_container);
        return nullptr;
    }

    jmethodID get_holder_method = env->GetMethodID(surface_view_class, "getHolder", "()Landroid/view/SurfaceHolder;");
    log_jni_exception(env, "Could not get SurfaceView.getHolder");
    env->DeleteLocalRef(surface_view_class);
    if (!get_holder_method) {
        env->DeleteLocalRef(surface_container);
        return nullptr;
    }

    jobject holder = env->CallObjectMethod(surface_container, get_holder_method);
    log_jni_exception(env, "Could not call SurfaceView.getHolder");
    env->DeleteLocalRef(surface_container);
    if (!holder)
        return nullptr;

    jclass holder_class = env->GetObjectClass(holder);
    log_jni_exception(env, "Could not get SurfaceHolder class");
    if (!holder_class) {
        env->DeleteLocalRef(holder);
        return nullptr;
    }

    jmethodID get_surface_method = env->GetMethodID(holder_class, "getSurface", "()Landroid/view/Surface;");
    log_jni_exception(env, "Could not get SurfaceHolder.getSurface");
    env->DeleteLocalRef(holder_class);
    if (!get_surface_method) {
        env->DeleteLocalRef(holder);
        return nullptr;
    }

    jobject surface = env->CallObjectMethod(holder, get_surface_method);
    log_jni_exception(env, "Could not call SurfaceHolder.getSurface");
    env->DeleteLocalRef(holder);
    if (!surface)
        return nullptr;

    ANativeWindow* native_window = ANativeWindow_fromSurface(env, surface);
    env->DeleteLocalRef(surface);
    return native_window;
}

bool Window::create_webgpu_surface()
{
    const VkSurfaceKHR qt_surface = QVulkanInstance::surfaceForWindow(this);
    if (qt_surface == VK_NULL_HANDLE)
        qWarning() << "Qt did not create a Vulkan surface for the window";

    m_native_window = android_native_window();
    if (!m_native_window)
        return false;

    WGPUSurfaceSourceAndroidNativeWindow android_window_desc {};
    android_window_desc.chain.next = nullptr;
    android_window_desc.chain.sType = WGPUSType_SurfaceSourceAndroidNativeWindow;
    android_window_desc.window = m_native_window;

    WGPUSurfaceDescriptor surface_desc {};
    surface_desc.nextInChain = &android_window_desc.chain;
    surface_desc.label = WGPUStringView { .data = "Qt Android surface", .length = WGPU_STRLEN };

    m_surface = wgpuInstanceCreateSurface(m_instance, &surface_desc);
    return m_surface != nullptr;
}

void Window::create_webgpu_context()
{
    WGPUInstanceFeatureName timed_wait_feature = WGPUInstanceFeatureName_TimedWaitAny;
    WGPUInstanceDescriptor instance_desc {};
    instance_desc.requiredFeatureCount = 1;
    instance_desc.requiredFeatures = &timed_wait_feature;
    m_instance = wgpuCreateInstance(&instance_desc);
    if (!m_instance)
        qFatal("Could not create WebGPU instance");

    if (!create_webgpu_surface())
        qFatal("Could not create WebGPU surface from Qt Android window");

    WGPURequestAdapterOptions adapter_opts {};
    adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;
    adapter_opts.backendType = WGPUBackendType_Vulkan;
    adapter_opts.compatibleSurface = m_surface;
    m_adapter = webgpu::requestAdapterSync(m_instance, adapter_opts);
    if (!m_adapter)
        qFatal("Could not get Vulkan WebGPU adapter");

    WGPULimits supported_limits {};
    wgpuAdapterGetLimits(m_adapter, &supported_limits);

    WGPULimits required_limits {};
    required_limits.minStorageBufferOffsetAlignment = supported_limits.minStorageBufferOffsetAlignment;
    required_limits.minUniformBufferOffsetAlignment = supported_limits.minUniformBufferOffsetAlignment;
    required_limits.maxInterStageShaderVariables = WGPU_LIMIT_U32_UNDEFINED;
    required_limits.maxBindGroups = std::max(required_limits.maxBindGroups, 3u);

    constexpr uint32_t required_color_attachment_bytes_per_sample = 32u;
    if (supported_limits.maxColorAttachmentBytesPerSample < required_color_attachment_bytes_per_sample)
        qFatal("WebGPU adapter does not support the tile renderer color attachment layout");
    required_limits.maxColorAttachmentBytesPerSample = required_color_attachment_bytes_per_sample;

    constexpr uint32_t tile_array_layers = 512u;
    if (supported_limits.maxTextureArrayLayers < tile_array_layers)
        qFatal("WebGPU adapter does not support enough texture array layers for the plain renderer");
    required_limits.maxTextureArrayLayers = tile_array_layers;

    constexpr uint64_t desired_max_buffer_size = 512ull * 1024ull * 1024ull;
    required_limits.maxBufferSize = std::min(supported_limits.maxBufferSize, desired_max_buffer_size);

    WGPUDeviceDescriptor device_desc {};
    device_desc.label = WGPUStringView { .data = "plain_wgpu_renderer device", .length = WGPU_STRLEN };
    device_desc.requiredLimits = &required_limits;
    device_desc.defaultQueue.label = WGPUStringView { .data = "plain_wgpu_renderer queue", .length = WGPU_STRLEN };
    device_desc.uncapturedErrorCallbackInfo = WGPUUncapturedErrorCallbackInfo {
        .nextInChain = nullptr,
        .callback = webgpu_device_error_callback,
        .userdata1 = nullptr,
        .userdata2 = nullptr,
    };
    device_desc.deviceLostCallbackInfo = WGPUDeviceLostCallbackInfo {
        .nextInChain = nullptr,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = webgpu_device_lost_callback,
        .userdata1 = nullptr,
        .userdata2 = nullptr,
    };

    m_device = webgpu::requestDeviceSync(m_instance, m_adapter, device_desc);
    if (!m_device)
        qFatal("Could not get WebGPU device");

    m_queue = wgpuDeviceGetQueue(m_device);
    if (!m_queue)
        qFatal("Could not get WebGPU queue");

    m_webgpu_context.init(m_instance, m_device, m_adapter, m_surface, m_queue);
}

void Window::configure_surface(const glm::uvec2& size)
{
    if (size == m_surface_size)
        return;

    WGPUSurfaceCapabilities surface_capabilities {};
    wgpuSurfaceGetCapabilities(m_surface, m_adapter, &surface_capabilities);
    if (surface_capabilities.formatCount < 1)
        qFatal("WebGPU surface does not report any supported formats");

    m_surface_texture_format = surface_capabilities.formats[0];
    for (size_t i = 0; i < surface_capabilities.formatCount; ++i) {
        if (surface_capabilities.formats[i] == WGPUTextureFormat_BGRA8Unorm || surface_capabilities.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
            m_surface_texture_format = surface_capabilities.formats[i];
            break;
        }
    }
    m_webgpu_context.set_surface_texture_format(m_surface_texture_format);

    WGPUSurfaceConfiguration config {};
    config.width = size.x;
    config.height = size.y;
    config.format = m_surface_texture_format;
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.device = m_device;
    config.presentMode = m_surface_present_mode;
    config.alphaMode = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(m_surface, &config);
    wgpuSurfaceCapabilitiesFreeMembers(surface_capabilities);

    m_surface_size = size;
}

void Window::create_buffers()
{
    m_shared_config_ubo = std::make_unique<webgpu::Buffer<webgpu_engine::uboSharedConfig>>(m_device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
    m_camera_config_ubo = std::make_unique<webgpu::Buffer<webgpu_engine::uboCameraConfig>>(m_device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
}

void Window::create_bind_groups()
{
    m_shared_config_bind_group = std::make_unique<webgpu::raii::BindGroup>(m_device,
        m_context->webgpu_ctx().resource_registry().bind_group_layout("shared_config"),
        std::initializer_list<WGPUBindGroupEntry> { m_shared_config_ubo->raw_buffer().create_bind_group_entry(0) });

    m_camera_bind_group = std::make_unique<webgpu::raii::BindGroup>(m_device,
        m_context->webgpu_ctx().resource_registry().bind_group_layout("camera"),
        std::initializer_list<WGPUBindGroupEntry> { m_camera_config_ubo->raw_buffer().create_bind_group_entry(0) });
}

void Window::create_compose_pipeline()
{
    WGPUBindGroupLayoutEntry albedo_entry {};
    albedo_entry.binding = 0;
    albedo_entry.visibility = WGPUShaderStage_Fragment;
    albedo_entry.texture.sampleType = WGPUTextureSampleType_Uint;
    albedo_entry.texture.viewDimension = WGPUTextureViewDimension_2D;
    m_compose_bind_group_layout
        = std::make_unique<webgpu::raii::BindGroupLayout>(m_device, std::vector<WGPUBindGroupLayoutEntry> { albedo_entry }, "plain compose layout");

    static constexpr char compose_shader[] = R"(
@group(0) @binding(0) var albedo_texture: texture_2d<u32>;

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) texcoords: vec2f,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    const vertices = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var out: VertexOut;
    out.position = vec4f(vertices[vertex_index], 0.0, 1.0);
    out.texcoords = vec2f(0.5, -0.5) * out.position.xy + vec2f(0.5);
    return out;
}

@fragment
fn fragmentMain(vertex_out: VertexOut) -> @location(0) vec4f {
    let dims = textureDimensions(albedo_texture);
    let tci = min(vec2u(vertex_out.texcoords * vec2f(dims)), dims - vec2u(1, 1));
    return unpack4x8unorm(textureLoad(albedo_texture, tci, 0).r);
}
)";

    m_compose_shader = m_context->webgpu_ctx().resource_registry().compile_shader_from_code(m_device, compose_shader, "plain compose shader");

    webgpu::FramebufferFormat format {};
    format.depth_format = WGPUTextureFormat_Undefined;
    format.color_formats.emplace_back(m_surface_texture_format);

    m_compose_pipeline = std::make_unique<webgpu::raii::GenericRenderPipeline>(m_device,
        *m_compose_shader,
        *m_compose_shader,
        std::vector<webgpu::util::SingleVertexBufferInfo> {},
        format,
        std::vector<const webgpu::raii::BindGroupLayout*> { m_compose_bind_group_layout.get() });
}

void Window::resize_framebuffer(const glm::uvec2& size)
{
    webgpu::FramebufferFormat format = m_context->tile_mesh_renderer()->render_tiles_pipeline().framebuffer_format();
    format.size = size;
    m_gbuffer = std::make_unique<webgpu::Framebuffer>(m_device, format);

    m_compose_bind_group = std::make_unique<webgpu::raii::BindGroup>(
        m_device, *m_compose_bind_group_layout, std::initializer_list<WGPUBindGroupEntry> { m_gbuffer->color_texture_view(0).create_bind_group_entry(0) });
}

void Window::render()
{
    m_render_queued = false;
    if (!m_initialized || m_destroyed || !isExposed())
        return;

    WGPUSurfaceTexture surface_texture {};
    wgpuSurfaceGetCurrentTexture(m_surface, &surface_texture);
    if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
        && surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        qWarning() << "Could not get current WebGPU surface texture:" << surface_texture.status;
        return;
    }

    WGPUTextureViewDescriptor view_desc {};
    view_desc.label = WGPUStringView { .data = "surface texture view", .length = WGPU_STRLEN };
    view_desc.format = wgpuTextureGetFormat(surface_texture.texture);
    view_desc.dimension = WGPUTextureViewDimension_2D;
    view_desc.mipLevelCount = 1;
    view_desc.arrayLayerCount = 1;
    view_desc.aspect = WGPUTextureAspect_All;
    WGPUTextureView surface_texture_view = wgpuTextureCreateView(surface_texture.texture, &view_desc);
    if (!surface_texture_view)
        qFatal("Cannot create surface texture view");

    WGPUCommandEncoderDescriptor encoder_desc {};
    encoder_desc.label = WGPUStringView { .data = "plain_wgpu_renderer encoder", .length = WGPU_STRLEN };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encoder_desc);

    m_shared_config_ubo->data = m_context->shared_config();
    m_shared_config_ubo->update_gpu_data(m_queue);

    {
        std::unique_ptr<webgpu::raii::RenderPassEncoder> render_pass = m_gbuffer->begin_render_pass(encoder);
        wgpuRenderPassEncoderSetBindGroup(render_pass->handle(), 0, m_shared_config_bind_group->handle(), 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(render_pass->handle(), 1, m_camera_bind_group->handle(), 0, nullptr);

        using namespace nucleus::tile;
        const auto draw_list
            = drawing::compute_bounds(drawing::limit(drawing::generate_list(m_camera, m_context->aabb_decorator(), 18), 512), m_context->aabb_decorator());
        const auto culled_draw_list = drawing::sort(drawing::cull(draw_list, m_camera), m_camera.position());
        m_context->tile_mesh_renderer()->draw(render_pass->handle(), m_camera, culled_draw_list);
    }

    {
        webgpu::raii::RenderPassEncoder render_pass(encoder, surface_texture_view, nullptr);
        wgpuRenderPassEncoderSetPipeline(render_pass.handle(), m_compose_pipeline->pipeline().handle());
        wgpuRenderPassEncoderSetBindGroup(render_pass.handle(), 0, m_compose_bind_group->handle(), 0, nullptr);
        wgpuRenderPassEncoderDraw(render_pass.handle(), 3, 1, 0, 0);
    }

    wgpuTextureViewRelease(surface_texture_view);

    WGPUCommandBufferDescriptor command_desc {};
    command_desc.label = WGPUStringView { .data = "plain_wgpu_renderer command buffer", .length = WGPU_STRLEN };
    WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, &command_desc);
    wgpuCommandEncoderRelease(encoder);
    wgpuQueueSubmit(m_queue, 1, &command);
    wgpuCommandBufferRelease(command);
    wgpuSurfacePresent(m_surface);
    wgpuDeviceTick(m_device);
}

void Window::release_webgpu_context()
{
    if (m_surface)
        wgpuSurfaceUnconfigure(m_surface);
    if (m_queue)
        wgpuQueueRelease(m_queue);
    if (m_surface)
        wgpuSurfaceRelease(m_surface);
    if (m_device)
        wgpuDeviceRelease(m_device);
    if (m_adapter)
        wgpuAdapterRelease(m_adapter);
    if (m_instance)
        wgpuInstanceRelease(m_instance);
    if (m_native_window)
        ANativeWindow_release(m_native_window);

    m_queue = nullptr;
    m_surface = nullptr;
    m_device = nullptr;
    m_adapter = nullptr;
    m_instance = nullptr;
    m_native_window = nullptr;
}
