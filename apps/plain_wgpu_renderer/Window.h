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

#pragma once

#include "nucleus/camera/AbstractDepthTester.h"
#include "nucleus/camera/Definition.h"
#include "nucleus/event_parameter.h"
#include "webgpu/base/Buffer.h"
#include "webgpu/base/Context.h"
#include "webgpu/base/Framebuffer.h"
#include "webgpu/base/raii/BindGroup.h"
#include "webgpu/base/raii/BindGroupLayout.h"
#include "webgpu/base/raii/Pipeline.h"
#include "webgpu/engine/Context.h"
#include "webgpu/engine/UniformBufferObjects.h"

#include <QTimer>
#include <QVulkanInstance>
#include <QWindow>
#include <memory>
#include <webgpu/webgpu.h>

class VulkanPresentation;

class Window : public QWindow, public nucleus::camera::AbstractDepthTester {
    Q_OBJECT
public:
    explicit Window(std::shared_ptr<webgpu_engine::Context> context);
    ~Window() override;

    void initialise_gpu();
    void destroy();

    [[nodiscard]] float depth(const glm::dvec2& normalised_device_coordinates) override;
    [[nodiscard]] glm::dvec3 position(const glm::dvec2& normalised_device_coordinates) override;

public slots:
    void update_camera(const nucleus::camera::Definition& new_definition);
    void request_render();

signals:
    void mouse_pressed(const nucleus::event_parameter::Mouse&) const;
    void mouse_moved(const nucleus::event_parameter::Mouse&) const;
    void wheel_turned(const nucleus::event_parameter::Wheel&) const;
    void touch_made(const nucleus::event_parameter::Touch&) const;
    void key_pressed(const QKeyCombination&) const;
    void key_released(const QKeyCombination&) const;
    void resized(const glm::uvec2&) const;
    void about_to_be_destoryed() const;
    void initialisation_started() const;

protected:
    void exposeEvent(QExposeEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void touchEvent(QTouchEvent* event) override;

private:
    [[nodiscard]] glm::uvec2 logical_size() const;
    [[nodiscard]] glm::uvec2 pixel_size() const;

    void create_webgpu_context();
    void configure_presentation(const glm::uvec2& size);
    void create_buffers();
    void create_bind_groups();
    void create_compose_pipeline();
    void resize_framebuffer(const glm::uvec2& size);
    void render();
    void release_webgpu_context();

private:
    std::shared_ptr<webgpu_engine::Context> m_context;
    webgpu::Context m_webgpu_context;
    QVulkanInstance m_vulkan_instance;
    QTimer m_render_timer;
    std::unique_ptr<VulkanPresentation> m_presentation;

    WGPUInstance m_instance = nullptr;
    WGPUAdapter m_adapter = nullptr;
    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;
    WGPUTextureFormat m_surface_texture_format = WGPUTextureFormat_BGRA8Unorm;

    std::unique_ptr<webgpu::Buffer<webgpu_engine::uboSharedConfig>> m_shared_config_ubo;
    std::unique_ptr<webgpu::Buffer<webgpu_engine::uboCameraConfig>> m_camera_config_ubo;
    std::unique_ptr<webgpu::raii::BindGroup> m_shared_config_bind_group;
    std::unique_ptr<webgpu::raii::BindGroup> m_camera_bind_group;
    std::unique_ptr<webgpu::raii::BindGroupLayout> m_compose_bind_group_layout;
    std::unique_ptr<webgpu::raii::BindGroup> m_compose_bind_group;
    std::unique_ptr<webgpu::raii::GenericRenderPipeline> m_compose_pipeline;
    std::unique_ptr<webgpu::raii::ShaderModule> m_compose_shader;
    std::unique_ptr<webgpu::Framebuffer> m_gbuffer;

    nucleus::camera::Definition m_camera;
    glm::uvec2 m_surface_size = glm::uvec2(0);
    bool m_initialized = false;
    bool m_render_queued = false;
    bool m_destroyed = false;
};
