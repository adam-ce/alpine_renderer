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

#include <glm/glm.hpp>
#include <memory>
#include <webgpu/webgpu.h>

class QVulkanInstance;
class QWindow;

class VulkanPresentation {
public:
    VulkanPresentation(QVulkanInstance& vulkan_instance, QWindow& window);
    ~VulkanPresentation();

    VulkanPresentation(const VulkanPresentation&) = delete;
    VulkanPresentation& operator=(const VulkanPresentation&) = delete;

    void initialize(WGPUAdapter adapter, WGPUDevice device, const glm::uvec2& size);
    void resize(const glm::uvec2& size);
    void destroy();

    [[nodiscard]] WGPUTextureFormat texture_format() const;
    [[nodiscard]] glm::uvec2 size() const;
    [[nodiscard]] WGPUTextureView render_target_view() const;

    void begin_dawn_access();
    void end_dawn_access_and_present();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
