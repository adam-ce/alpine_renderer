/*****************************************************************************
 * Alpine Renderer
 * Copyright (C) 2022 Adam Celarek
 * Copyright (C) 2023 Jakob Lindner
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
#include <vector>

#include <QImage>
#include <glm/glm.hpp>
#include <QOpenGLTexture>
#include <QColor>


// There is QOpenGLFramebufferObject, but its API lacks the following:
// - get depth buffer as a texture
// - resize (while keeping all parametres)
// - limited depth / colour formats (to ease the pain of managing)


namespace gl_engine {

struct TextureDefinition;

class Framebuffer
{
public:
    enum class DepthFormat {
        None,
        Int16,
        Int24,
        Float32
    };
    enum class ColourFormat {
        R8,
        R16UI,
        RGBA8,
        RGBA16UI,
        RGB16F,         // NOT COLOR RENDERABLE ON OPENGLES
        RGBA16F,        // NOT COLOR RENDERABLE ON OPENGLES
        Float32         // NOT COLOR RENDERABLE ON OPENGLES
    };

private:
    DepthFormat m_depth_format;
    std::vector<TextureDefinition> m_colour_definitions;

    std::unique_ptr<QOpenGLTexture> m_depth_texture;
    std::vector<std::unique_ptr<QOpenGLTexture>> m_colour_textures;
    //std::unique_ptr<QOpenGLTexture> m_colour_texture;
    unsigned m_frame_buffer = -1;
    glm::uvec2 m_size;
    // Recreates the OpenGL-Texture for the given index. An index of -1 recreates the depth-buffer.
    void recreate_texture(int index);
    // Calls recreate_texture for all the buffers that are attached to this FBO (depth and colour)
    void recreate_all_textures();

public:
    Framebuffer(DepthFormat depth_format, std::vector<TextureDefinition> colour_definitions, glm::uvec2 init_size = {4,4});
    ~Framebuffer();
    void resize(const glm::uvec2& new_size);
    void bind();
    void bind_colour_texture(unsigned index = 0, unsigned location = 0);

    [[deprecated("Not in use, untested...")]]
    std::unique_ptr<QOpenGLTexture> take_and_replace_colour_attachment(unsigned index);

    QImage read_colour_attachment(unsigned index);
    std::array<uchar, 4> read_colour_attachment_pixel(unsigned index, const glm::dvec2& normalised_device_coordinates);
    static void unbind();

    glm::uvec2 size() const;

private:
    void reset_fbo();
};

struct TextureDefinition {
    Framebuffer::ColourFormat format = Framebuffer::ColourFormat::RGBA8;
    QOpenGLTexture::Filter minFilter = QOpenGLTexture::Filter::Linear;
    QOpenGLTexture::Filter magFilter = QOpenGLTexture::Filter::Linear;
    QOpenGLTexture::WrapMode wrapMode = QOpenGLTexture::WrapMode::ClampToEdge;
    QColor borderColor = QColor(0.0f, 0.0f, 0.0f, 1.0f);
    bool autoMipMapGeneration = false;
};

}
