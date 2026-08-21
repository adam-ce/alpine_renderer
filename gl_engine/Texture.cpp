/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2024 Adam Celarek
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

#include "Texture.h"
#include "ShaderProgram.h"
#include "nucleus/utils/ColourTexture.h"

#include <QOpenGLExtraFunctions>
#include <QOpenGLFunctions>
#include <QtAssert>
#include <algorithm>
#include <array>
#include <cmath>
#ifdef __EMSCRIPTEN__
#include <emscripten/html5_webgl.h>
#endif
#ifdef ANDROID
#include <GLES3/gl3.h>
#endif

namespace {
struct GlParams {
    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    unsigned n_elements = 0;
    unsigned n_bytes_per_element = 0;
    bool is_texture_filterable = false;
};

// https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexImage2D.xhtml
GlParams gl_tex_params(gl_engine::Texture::Format format)
{
    using F = gl_engine::Texture::Format;
    switch (format) {
    case F::CompressedRGBA8:
        return { GLint(gl_engine::Texture::compressed_texture_format()), 0, 0, 0, 0, true };
    case F::RGBA8:
        return { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 1, true };
    case F::SRGBA8:
        return { GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 1, true };
    case F::RGBA8UI:
        return { GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, 4, 1 };
    case F::RGBA32F:
        return { GL_RGBA32F, GL_RGBA, GL_FLOAT, 4, 4 };
    case F::RG8:
        return { GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2, 1, true };
    case F::RG32UI:
        return { GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT, 2, 4 };
    case F::RGB32UI:
        return { GL_RGB32UI, GL_RGB_INTEGER, GL_UNSIGNED_INT, 3, 4 };
    case F::R8UI:
        return { GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 1, 1 };
    case F::R16UI:
        return { GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT, 1, 2 };
    case F::R32UI:
        return { GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1, 4 };
    case F::Invalid:
        return {};
    }
    return {};
}
} // namespace

gl_engine::Texture::Texture(Target target, Format format)
    : m_target(target)
    , m_format(format)
{
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    f->glGenTextures(1, &m_id);
}

gl_engine::Texture::~Texture()
{
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    f->glDeleteTextures(1, &m_id);
}

void gl_engine::Texture::bind(unsigned int texture_unit)
{
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    f->glActiveTexture(GL_TEXTURE0 + texture_unit);
    f->glBindTexture(GLenum(m_target), m_id);
}

void gl_engine::Texture::setParams(Filter min_filter, Filter mag_filter, bool anisotropic_filtering)
{
    // doesn't make sense, does it?
    Q_ASSERT(mag_filter != Filter::MipMapLinear);

    Q_ASSERT(gl_tex_params(m_format).is_texture_filterable || (min_filter == Filter::Nearest && mag_filter == Filter::Nearest));

    m_min_filter = min_filter;
    m_mag_filter = mag_filter;

    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    f->glBindTexture(GLenum(m_target), m_id);
    f->glTexParameteri(GLenum(m_target), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    f->glTexParameteri(GLenum(m_target), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    f->glTexParameteri(GLenum(m_target), GL_TEXTURE_MIN_FILTER, GLint(m_min_filter));
    f->glTexParameteri(GLenum(m_target), GL_TEXTURE_MAG_FILTER, GLint(m_mag_filter));
    if (anisotropic_filtering && max_anisotropy() > 0)
        f->glTexParameterf(GLenum(m_target), max_anisotropy_param(), max_anisotropy());
}

void gl_engine::Texture::allocate_array(unsigned int width, unsigned int height, unsigned int n_layers)
{
    Q_ASSERT(m_target == Target::_2dArray);
    Q_ASSERT(m_format != Format::Invalid);

    auto mip_level_count = 1;
    if (m_min_filter == Filter::MipMapLinear)
        mip_level_count = GLsizei(1 + std::floor(std::log2(std::max(width, height))));

    m_width = width;
    m_height = height;
    m_n_layers = n_layers;

    auto* f = QOpenGLContext::currentContext()->extraFunctions();
    f->glBindTexture(GLenum(m_target), m_id);
    f->glTexStorage3D(GLenum(m_target), mip_level_count, gl_tex_params(m_format).internal_format, GLsizei(width), GLsizei(height), GLsizei(n_layers));
}

void gl_engine::Texture::upload(const nucleus::utils::ColourTexture& texture)
{
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    f->glBindTexture(GLenum(m_target), m_id);
    f->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    const auto width = GLsizei(texture.width());
    const auto height = GLsizei(texture.height());
    const auto p = gl_tex_params(m_format);
    if (m_format == Format::CompressedRGBA8) {
        Q_ASSERT(m_min_filter != Filter::MipMapLinear);
        const auto format = gl_engine::Texture::compressed_texture_format();
        f->glCompressedTexImage2D(GLenum(m_target), 0, format, width, height, 0, GLsizei(texture.n_bytes()), texture.data());
    } else if (m_format == Format::RGBA8 || m_format == Format::SRGBA8) {
        f->glTexImage2D(GLenum(m_target), 0, p.internal_format, width, height, 0, p.format, p.type, texture.data());
        if (m_min_filter == Filter::MipMapLinear)
            f->glGenerateMipmap(GLenum(m_target));
    } else {
        Q_ASSERT(false);
    }
}

void gl_engine::Texture::upload(const nucleus::utils::ColourTexture& texture, unsigned int array_index)
{
    Q_ASSERT(texture.width() == m_width);
    Q_ASSERT(texture.height() == m_height);
    Q_ASSERT(array_index < m_n_layers);
    Q_ASSERT(m_min_filter != Filter::MipMapLinear); // use the upload function with nucleus::utils::MipmappedColourTexture

    auto* f = QOpenGLContext::currentContext()->extraFunctions();
    f->glBindTexture(GLenum(m_target), m_id);
    f->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    const auto width = GLsizei(texture.width());
    const auto height = GLsizei(texture.height());
    if (m_format == Format::CompressedRGBA8) {
        const auto format = gl_engine::Texture::compressed_texture_format();
        f->glCompressedTexSubImage3D(GLenum(m_target), 0, 0, 0, GLint(array_index), width, height, 1, format, GLsizei(texture.n_bytes()), texture.data());
    } else if (m_format == Format::RGBA8 || m_format == Format::SRGBA8) {
        f->glTexSubImage3D(GLenum(m_target), 0, 0, 0, GLint(array_index), width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE, texture.data());
    } else {
        Q_ASSERT(false);
    }
}

void gl_engine::Texture::upload(const nucleus::utils::MipmappedColourTexture& mipped_texture, unsigned int array_index)
{
    Q_ASSERT(mipped_texture.size() > 0);
    Q_ASSERT(mipped_texture.front().width() == m_width);
    Q_ASSERT(mipped_texture.front().height() == m_height);
    Q_ASSERT(array_index < m_n_layers);

    auto* f = QOpenGLContext::currentContext()->extraFunctions();
    f->glBindTexture(GLenum(m_target), m_id);
    f->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    auto mip_level = 0;
    for (const auto& texture : mipped_texture) {
        const auto width = GLsizei(texture.width());
        const auto height = GLsizei(texture.height());
        if (m_format == Format::CompressedRGBA8) {
            const auto format = gl_engine::Texture::compressed_texture_format();
            f->glCompressedTexSubImage3D(
                GLenum(m_target), mip_level, 0, 0, GLint(array_index), width, height, 1, format, GLsizei(texture.n_bytes()), texture.data());
        } else if (m_format == Format::RGBA8 || m_format == Format::SRGBA8) {
            f->glTexSubImage3D(GLenum(m_target), mip_level, 0, 0, GLint(array_index), width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE, texture.data());
        } else {
            Q_ASSERT(false);
        }
        ++mip_level;
    }
}

template <typename T> void gl_engine::Texture::upload(const radix::Raster<T>& texture, unsigned int array_index)
{
    Q_ASSERT(m_target == Target::_2dArray);

    const auto p = gl_tex_params(m_format);
    Q_ASSERT(m_format != Format::CompressedRGBA8);
    Q_ASSERT(m_format != Format::Invalid);
    Q_ASSERT(sizeof(T) == p.n_bytes_per_element * p.n_elements);
    if (!p.is_texture_filterable) {
        Q_ASSERT(m_mag_filter == Filter::Nearest);
        Q_ASSERT(m_min_filter == Filter::Nearest);
    }
    Q_ASSERT(array_index < m_n_layers);
    Q_ASSERT(texture.width() == m_width);
    Q_ASSERT(texture.height() == m_height);

    const auto width = GLsizei(texture.width());
    const auto height = GLsizei(texture.height());

    auto* f = QOpenGLContext::currentContext()->extraFunctions();
    f->glBindTexture(GLenum(m_target), m_id);
    f->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    f->glTexSubImage3D(GLenum(m_target), 0, 0, 0, GLint(array_index), width, height, 1, p.format, p.type, texture.bytes().data());

    if (m_min_filter == Filter::MipMapLinear)
        f->glGenerateMipmap(GLenum(m_target));
}
template void gl_engine::Texture::upload<uint8_t>(const radix::Raster<uint8_t>&, unsigned);
template void gl_engine::Texture::upload<uint16_t>(const radix::Raster<uint16_t>&, unsigned);
template void gl_engine::Texture::upload<uint32_t>(const radix::Raster<uint32_t>&, unsigned);
template void gl_engine::Texture::upload<glm::vec<2, uint8_t>>(const radix::Raster<glm::vec<2, uint8_t>>&, unsigned);
template void gl_engine::Texture::upload<glm::vec<2, uint32_t>>(const radix::Raster<glm::vec<2, uint32_t>>&, unsigned);
template void gl_engine::Texture::upload<glm::vec<3, uint32_t>>(const radix::Raster<glm::vec<3, uint32_t>>&, unsigned);
template void gl_engine::Texture::upload<glm::vec<4, uint8_t>>(const radix::Raster<glm::vec<4, uint8_t>>&, unsigned);
template void gl_engine::Texture::upload<glm::vec<4, float>>(const radix::Raster<glm::vec<4, float>>&, unsigned);

template <typename T> void gl_engine::Texture::upload(const radix::Raster<T>& texture)
{
    Q_ASSERT(m_target == Target::_2d);

    const auto p = gl_tex_params(m_format);
    Q_ASSERT(m_format != Format::CompressedRGBA8);
    Q_ASSERT(m_format != Format::Invalid);
    Q_ASSERT(sizeof(T) == p.n_bytes_per_element * p.n_elements);
    if (!p.is_texture_filterable) {
        Q_ASSERT(m_mag_filter == Filter::Nearest);
        Q_ASSERT(m_min_filter == Filter::Nearest);
    }

    QOpenGLExtraFunctions* f = QOpenGLContext::currentContext()->extraFunctions();
    f->glBindTexture(GLenum(m_target), m_id);
    f->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    f->glTexImage2D(GLenum(m_target), 0, p.internal_format, GLsizei(texture.width()), GLsizei(texture.height()), 0, p.format, p.type, texture.bytes().data());

    if (m_min_filter == Filter::MipMapLinear)
        f->glGenerateMipmap(GLenum(m_target));
}
template void gl_engine::Texture::upload<uint8_t>(const radix::Raster<uint8_t>&);
template void gl_engine::Texture::upload<uint16_t>(const radix::Raster<uint16_t>&);
template void gl_engine::Texture::upload<uint32_t>(const radix::Raster<uint32_t>&);
template void gl_engine::Texture::upload<glm::vec<2, uint32_t>>(const radix::Raster<glm::vec<2, uint32_t>>&);
template void gl_engine::Texture::upload<glm::vec<3, uint32_t>>(const radix::Raster<glm::vec<3, uint32_t>>&);
template void gl_engine::Texture::upload<glm::vec<2, uint8_t>>(const radix::Raster<glm::vec<2, uint8_t>>&);
template void gl_engine::Texture::upload<glm::vec<4, uint8_t>>(const radix::Raster<glm::vec<4, uint8_t>>&);
template void gl_engine::Texture::upload<glm::vec<4, float>>(const radix::Raster<glm::vec<4, float>>&);

GLenum gl_engine::Texture::compressed_texture_format()
{
    // select between
    // DXT1, also called s3tc, old desktop compression
    // ETC1, old mobile compression
#if defined(__EMSCRIPTEN__)
    static const GLenum gl_texture_format = []() {
        const auto context = emscripten_webgl_get_current_context();
        if (!context)
            qFatal("No current WebGL context while detecting texture compression");
        if (emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_etc"))
            return GLenum(GL_COMPRESSED_SRGB8_ETC2);

        const bool s3tc = emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_s3tc");
        const bool s3tc_srgb = emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_s3tc_srgb");
        if (!s3tc || !s3tc_srgb)
            qFatal("Neither ETC nor sRGB S3TC texture compression is supported");
        return GLenum(GL_COMPRESSED_SRGB_S3TC_DXT1_EXT);
    }();
    return gl_texture_format;
#elif defined(__ANDROID__)
    return GL_COMPRESSED_SRGB8_ETC2;
#else
    return GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
#endif
}

nucleus::utils::ColourTexture::Format gl_engine::Texture::compression_algorithm()
{
#if defined(__EMSCRIPTEN__)
    static const auto compression_algorithm = []() {
        const auto context = emscripten_webgl_get_current_context();
        if (!context)
            qFatal("No current WebGL context while detecting texture compression");
        if (emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_etc"))
            return nucleus::utils::ColourTexture::Format::ETC1;

        const bool s3tc = emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_s3tc");
        const bool s3tc_srgb = emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_s3tc_srgb");
        if (!s3tc || !s3tc_srgb)
            qFatal("Neither ETC nor sRGB S3TC texture compression is supported");
        return nucleus::utils::ColourTexture::Format::DXT1;
    }();
    return compression_algorithm;
#elif defined(__ANDROID__)
    return nucleus::utils::ColourTexture::Format::ETC1;
#else
    return nucleus::utils::ColourTexture::Format::DXT1;
#endif
}

GLenum gl_engine::Texture::max_anisotropy_param()
{
#if defined(__EMSCRIPTEN__)
    static const GLenum param = []() {
        const auto context = emscripten_webgl_get_current_context();
        if (!context)
            qFatal("No current WebGL context while detecting anisotropic filtering");
        return emscripten_webgl_enable_extension(context, "EXT_texture_filter_anisotropic") ? GLenum(GL_TEXTURE_MAX_ANISOTROPY_EXT) : GLenum(0);
    }();
    return param;
#elif defined(__ANDROID__)
    return GL_TEXTURE_MAX_ANISOTROPY_EXT;
#else
    return GL_TEXTURE_MAX_ANISOTROPY;
#endif
}

float gl_engine::Texture::max_anisotropy()
{
#if defined(__EMSCRIPTEN__)
    static const float max_anisotropy = []() {
        const auto context = emscripten_webgl_get_current_context();
        if (!context)
            qFatal("No current WebGL context while detecting anisotropic filtering");
        if (!emscripten_webgl_enable_extension(context, "EXT_texture_filter_anisotropic"))
            return 0.f;

        GLfloat value = 0.f;
        QOpenGLContext::currentContext()->extraFunctions()->glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &value);
        return std::min(32.f, value);
    }();
    return max_anisotropy;
#elif defined(__ANDROID__)
    static const float max_anisotropy = []() {
        if (QOpenGLContext::currentContext()->hasExtension("GL_EXT_texture_filter_anisotropic")) {
            QOpenGLExtraFunctions* f = QOpenGLContext::currentContext()->extraFunctions();
            GLfloat t = 0.0f;
            f->glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &t);
            return std::min(32.f, t);
        }
        qDebug() << "GL_EXT_texture_filter_anisotropic not present";
        qDebug() << "present extensions: ";
        for (const auto& e : QOpenGLContext::currentContext()->extensions()) {
            qDebug() << e;
        }
        return 0.f;
    }();
    return max_anisotropy;
#else
    static const float max_anisotropy = []() {
        QOpenGLExtraFunctions* f = QOpenGLContext::currentContext()->extraFunctions();
        GLfloat t = 0.0f;
        f->glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &t);
        return std::min(32.f, t);
    }();
    return max_anisotropy;
#endif
}

struct gl_engine::TextureCompressor::Impl {
    static constexpr unsigned max_shader_mip_levels = 16;

    unsigned width = 0;
    unsigned height = 0;
    unsigned max_batch_size = 0;
    unsigned scratch_layers = 0;
    GLsizei block_atlas_width = 0;
    GLsizei block_atlas_height = 0;
    GLsizei output_atlas_width = 0;
    GLsizei output_atlas_height = 0;
    GLuint scratch_texture = 0;
    GLuint encoded_texture = 0;
    GLuint encoded_buffer = 0;
    GLuint vertex_array = 0;
    GLuint encoding_framebuffer = 0;
    GLuint packing_framebuffer = 0;
    GLuint packing_renderbuffer = 0;
    std::unique_ptr<ShaderProgram> dxt1_fragment_program;
    std::unique_ptr<ShaderProgram> etc1_fragment_program;
    std::unique_ptr<ShaderProgram> etc1_fast_fragment_program;
    std::unique_ptr<ShaderProgram> etc1_fast_split_fragment_program;
    std::unique_ptr<ShaderProgram> etc1_fast_split_fused_fragment_program;
    std::unique_ptr<ShaderProgram> etc1_fast_split_bounds_fragment_program;
    std::unique_ptr<ShaderProgram> packing_program;

    Impl(unsigned texture_width, unsigned texture_height, unsigned maximum_batch_size)
        : width(texture_width)
        , height(texture_height)
        , max_batch_size(maximum_batch_size)
    {
        Q_ASSERT(width > 0 && height > 0 && max_batch_size > 0);
        Q_ASSERT(TextureCompressor::mip_level_count(width, height) <= max_shader_mip_levels);
        auto* f = QOpenGLContext::currentContext()->extraFunctions();
        size_t maximum_size = 0;
        for (unsigned level = 0; level < TextureCompressor::mip_level_count(width, height); ++level) {
            maximum_size += TextureCompressor::compressed_level_size(
                std::max(1u, width >> level), std::max(1u, height >> level));
        }
        maximum_size *= max_batch_size;

        GLint maximum_renderbuffer_size = 0;
        GLint maximum_texture_size = 0;
        f->glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maximum_renderbuffer_size);
        f->glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maximum_texture_size);
        const auto atlas_size = [](size_t pixels, GLint maximum_dimension) {
            const auto atlas_width = GLsizei(std::min({ pixels, size_t(maximum_dimension), size_t(256) }));
            const auto atlas_height = GLsizei((pixels + size_t(atlas_width) - 1) / size_t(atlas_width));
            Q_ASSERT(atlas_width > 0 && atlas_height > 0 && atlas_height <= maximum_dimension);
            return std::pair(atlas_width, atlas_height);
        };
        std::tie(block_atlas_width, block_atlas_height) = atlas_size(maximum_size / 8, maximum_texture_size);
        std::tie(output_atlas_width, output_atlas_height) = atlas_size(maximum_size / 4, maximum_renderbuffer_size);

        f->glGenBuffers(1, &encoded_buffer);
        f->glBindBuffer(GL_PIXEL_PACK_BUFFER, encoded_buffer);
        f->glBufferData(GL_PIXEL_PACK_BUFFER,
            GLsizeiptr(size_t(output_atlas_width) * size_t(output_atlas_height) * 4),
            nullptr,
            GL_STREAM_DRAW);
        f->glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        f->glGenVertexArrays(1, &vertex_array);

        GLint previous_draw_framebuffer = 0;
        GLint previous_read_framebuffer = 0;
        GLint previous_renderbuffer = 0;
        GLint previous_texture = 0;
        f->glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previous_draw_framebuffer);
        f->glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previous_read_framebuffer);
        f->glGetIntegerv(GL_RENDERBUFFER_BINDING, &previous_renderbuffer);
        f->glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous_texture);

        f->glGenTextures(1, &encoded_texture);
        f->glBindTexture(GL_TEXTURE_2D, encoded_texture);
        f->glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG32UI, block_atlas_width, block_atlas_height);
        f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        f->glGenFramebuffers(1, &encoding_framebuffer);
        f->glBindFramebuffer(GL_FRAMEBUFFER, encoding_framebuffer);
        f->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, encoded_texture, 0);
        Q_ASSERT(f->glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

        f->glGenFramebuffers(1, &packing_framebuffer);
        f->glGenRenderbuffers(1, &packing_renderbuffer);
        f->glBindRenderbuffer(GL_RENDERBUFFER, packing_renderbuffer);
        // RGBA8UI with RGBA_INTEGER/UNSIGNED_BYTE is the portable WebGL 2 integer readback path.
        // Two pixels hold the two 32-bit words of each compressed block.
        f->glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8UI, output_atlas_width, output_atlas_height);
        f->glBindFramebuffer(GL_FRAMEBUFFER, packing_framebuffer);
        f->glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, packing_renderbuffer);
        Q_ASSERT(f->glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, GLuint(previous_draw_framebuffer));
        f->glBindFramebuffer(GL_READ_FRAMEBUFFER, GLuint(previous_read_framebuffer));
        f->glBindRenderbuffer(GL_RENDERBUFFER, GLuint(previous_renderbuffer));
        f->glBindTexture(GL_TEXTURE_2D, GLuint(previous_texture));

        dxt1_fragment_program = std::make_unique<ShaderProgram>("texture_compress_raster.vert",
            "texture_compress.vert",
            ShaderCodeSource::FILE);
        etc1_fragment_program = std::make_unique<ShaderProgram>("texture_compress_raster.vert",
            "texture_compress.vert",
            ShaderCodeSource::FILE,
            std::vector<QString> { QStringLiteral("#define ALP_COMPRESS_ETC1") });
        etc1_fast_fragment_program = std::make_unique<ShaderProgram>("texture_compress_raster.vert",
            "texture_compress.vert",
            ShaderCodeSource::FILE,
            std::vector<QString> { QStringLiteral("#define ALP_COMPRESS_ETC1"), QStringLiteral("#define ALP_COMPRESS_ETC1_FAST") });
        etc1_fast_split_fragment_program = std::make_unique<ShaderProgram>("texture_compress_raster.vert",
            "texture_compress.vert",
            ShaderCodeSource::FILE,
            std::vector<QString> { QStringLiteral("#define ALP_COMPRESS_ETC1"), QStringLiteral("#define ALP_COMPRESS_ETC1_SPLIT") });
        etc1_fast_split_fused_fragment_program = std::make_unique<ShaderProgram>("texture_compress_raster.vert",
            "texture_compress.vert",
            ShaderCodeSource::FILE,
            std::vector<QString> { QStringLiteral("#define ALP_COMPRESS_ETC1"), QStringLiteral("#define ALP_COMPRESS_ETC1_SPLIT_FUSED") });
        etc1_fast_split_bounds_fragment_program = std::make_unique<ShaderProgram>("texture_compress_raster.vert",
            "texture_compress.vert",
            ShaderCodeSource::FILE,
            std::vector<QString> { QStringLiteral("#define ALP_COMPRESS_ETC1"), QStringLiteral("#define ALP_COMPRESS_ETC1_SPLIT_BOUNDS") });
        packing_program = std::make_unique<ShaderProgram>(
            "texture_compress_raster.vert", "texture_compress_pack.frag", ShaderCodeSource::FILE);

    }

    ~Impl()
    {
        dxt1_fragment_program.reset();
        etc1_fragment_program.reset();
        etc1_fast_fragment_program.reset();
        etc1_fast_split_fragment_program.reset();
        etc1_fast_split_fused_fragment_program.reset();
        etc1_fast_split_bounds_fragment_program.reset();
        packing_program.reset();
        if (!QOpenGLContext::currentContext())
            return;
        auto* f = QOpenGLContext::currentContext()->extraFunctions();
        f->glDeleteFramebuffers(1, &encoding_framebuffer);
        f->glDeleteFramebuffers(1, &packing_framebuffer);
        f->glDeleteRenderbuffers(1, &packing_renderbuffer);
        f->glDeleteTextures(1, &encoded_texture);
        f->glDeleteVertexArrays(1, &vertex_array);
        f->glDeleteBuffers(1, &encoded_buffer);
        if (scratch_texture)
            f->glDeleteTextures(1, &scratch_texture);
    }

    void ensure_scratch_storage(unsigned layers)
    {
        if (scratch_layers == layers)
            return;
        auto* f = QOpenGLContext::currentContext()->extraFunctions();
        if (scratch_texture)
            f->glDeleteTextures(1, &scratch_texture);
        f->glGenTextures(1, &scratch_texture);
        f->glBindTexture(GL_TEXTURE_2D_ARRAY, scratch_texture);
        f->glTexStorage3D(GL_TEXTURE_2D_ARRAY,
            GLsizei(TextureCompressor::mip_level_count(width, height)),
            GL_RGBA8,
            GLsizei(width),
            GLsizei(height),
            GLsizei(layers));
        f->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        f->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        f->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        f->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        scratch_layers = layers;
    }
};

gl_engine::TextureCompressor::TextureCompressor(unsigned width, unsigned height, unsigned max_batch_size)
    : m(std::make_unique<Impl>(width, height, max_batch_size))
{
}

gl_engine::TextureCompressor::~TextureCompressor() = default;

size_t gl_engine::TextureCompressor::compressed_level_size(unsigned width, unsigned height)
{
    return size_t(std::max(1u, (width + 3) / 4)) * std::max(1u, (height + 3) / 4) * 8;
}

unsigned gl_engine::TextureCompressor::mip_level_count(unsigned width, unsigned height)
{
    Q_ASSERT(width > 0 && height > 0);
    return 1u + unsigned(std::floor(std::log2(std::max(width, height))));
}

bool gl_engine::TextureCompressor::is_supported()
{
#if defined(__EMSCRIPTEN__)
    const auto context = emscripten_webgl_get_current_context();
    if (!context)
        return false;
    if (emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_etc"))
        return true;
    return emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_s3tc")
        && emscripten_webgl_enable_extension(context, "WEBGL_compressed_texture_s3tc_srgb");
#else
    return true;
#endif
}

gl_engine::TextureCompressor::Result gl_engine::TextureCompressor::compress(std::span<const radix::Raster<glm::u8vec4>> textures,
    Texture& destination,
    std::span<const unsigned> destination_layers,
    const Settings& settings)
{
    Q_ASSERT(is_supported());
    Q_ASSERT(!textures.empty());
    Q_ASSERT(textures.size() == destination_layers.size());
    Q_ASSERT(textures.size() <= m->max_batch_size);
    Q_ASSERT(destination.m_target == Texture::Target::_2dArray);
    Q_ASSERT(destination.m_format == Texture::Format::CompressedRGBA8);
    Q_ASSERT(destination.m_width == m->width && destination.m_height == m->height);
    Q_ASSERT(settings.algorithm == Texture::compression_algorithm());
    Q_ASSERT(settings.effort <= 10);
    for (size_t i = 0; i < textures.size(); ++i) {
        Q_ASSERT(unsigned(textures[i].width()) == m->width && unsigned(textures[i].height()) == m->height);
        Q_ASSERT(destination_layers[i] < destination.m_n_layers);
    }

    m->ensure_scratch_storage(unsigned(textures.size()));
    auto* f = QOpenGLContext::currentContext()->extraFunctions();
    Result result;
    result.mip_levels = settings.generate_mipmaps ? mip_level_count(m->width, m->height) : 1;

    {
        f->glBindTexture(GL_TEXTURE_2D_ARRAY, m->scratch_texture);
        f->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        for (size_t layer = 0; layer < textures.size(); ++layer) {
            f->glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                0,
                0,
                0,
                GLint(layer),
                GLsizei(m->width),
                GLsizei(m->height),
                1,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                textures[layer].bytes().data());
        }
    }
    if (settings.generate_mipmaps) {
        f->glBindTexture(GL_TEXTURE_2D_ARRAY, m->scratch_texture);
        f->glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    }

    std::vector<size_t> level_offsets;
    std::vector<int> level_offsets_blocks;
    std::vector<int> level_blocks_x;
    std::vector<int> level_blocks_y;
    level_offsets.reserve(result.mip_levels);
    level_offsets_blocks.reserve(result.mip_levels);
    level_blocks_x.reserve(result.mip_levels);
    level_blocks_y.reserve(result.mip_levels);
    size_t total_encoded_size = 0;
    for (unsigned level = 0; level < result.mip_levels; ++level) {
        level_offsets.push_back(total_encoded_size);
        const auto level_width = std::max(1u, m->width >> level);
        const auto level_height = std::max(1u, m->height >> level);
        level_offsets_blocks.push_back(int(total_encoded_size / 8));
        level_blocks_x.push_back(int(std::max(1u, (level_width + 3) / 4)));
        level_blocks_y.push_back(int(std::max(1u, (level_height + 3) / 4)));
        total_encoded_size += compressed_level_size(level_width, level_height) * textures.size();
    }
    result.encoded_bytes = total_encoded_size;

    GLint previous_draw_framebuffer = 0;
    GLint previous_viewport[4] = {};
    GLboolean previous_colour_mask[4] = {};
    GLboolean blend_enabled = GL_FALSE;
    GLboolean cull_enabled = GL_FALSE;
    GLboolean depth_enabled = GL_FALSE;
    GLboolean scissor_enabled = GL_FALSE;

    {
        auto* program = m->dxt1_fragment_program.get();
        if (settings.algorithm == nucleus::utils::ColourTexture::Format::ETC1) {
            if (settings.encoder == Encoder::FastRange)
                program = m->etc1_fast_fragment_program.get();
            else if (settings.encoder == Encoder::FastSplit)
                program = m->etc1_fast_split_fragment_program.get();
            else if (settings.encoder == Encoder::FastSplitFused)
                program = m->etc1_fast_split_fused_fragment_program.get();
            else if (settings.encoder == Encoder::FastSplitBounds)
                program = m->etc1_fast_split_bounds_fragment_program.get();
            else
                program = m->etc1_fragment_program.get();
        }
        f->glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previous_draw_framebuffer);
        f->glGetIntegerv(GL_VIEWPORT, previous_viewport);
        f->glGetBooleanv(GL_COLOR_WRITEMASK, previous_colour_mask);
        blend_enabled = f->glIsEnabled(GL_BLEND);
        cull_enabled = f->glIsEnabled(GL_CULL_FACE);
        depth_enabled = f->glIsEnabled(GL_DEPTH_TEST);
        scissor_enabled = f->glIsEnabled(GL_SCISSOR_TEST);

        f->glDisable(GL_BLEND);
        f->glDisable(GL_CULL_FACE);
        f->glDisable(GL_DEPTH_TEST);
        f->glDisable(GL_SCISSOR_TEST);
        f->glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m->encoding_framebuffer);
        f->glViewport(0, 0, m->block_atlas_width, m->block_atlas_height);
        program->bind();
        program->set_uniform("source_texture", 7);
        program->set_uniform("texture_width", int(m->width));
        program->set_uniform("texture_height", int(m->height));
        program->set_uniform("effort", int(settings.effort));
        program->set_uniform("atlas_width", int(m->block_atlas_width));
        program->set_uniform("total_blocks", int(total_encoded_size / 8));
        program->set_uniform("mip_levels", int(result.mip_levels));
        program->set_uniform_array("level_offsets", level_offsets_blocks);
        program->set_uniform_array("level_blocks_x", level_blocks_x);
        program->set_uniform_array("level_blocks_y", level_blocks_y);
        f->glActiveTexture(GL_TEXTURE7);
        f->glBindTexture(GL_TEXTURE_2D_ARRAY, m->scratch_texture);
        f->glBindVertexArray(m->vertex_array);
        f->glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    {
        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m->packing_framebuffer);
        f->glViewport(0, 0, m->output_atlas_width, m->output_atlas_height);
        m->packing_program->bind();
        m->packing_program->set_uniform("encoded_blocks", 6);
        m->packing_program->set_uniform("block_atlas_width", int(m->block_atlas_width));
        m->packing_program->set_uniform("output_atlas_width", int(m->output_atlas_width));
        m->packing_program->set_uniform("total_blocks", int(total_encoded_size / 8));
        f->glActiveTexture(GL_TEXTURE6);
        f->glBindTexture(GL_TEXTURE_2D, m->encoded_texture);
        f->glDrawArrays(GL_TRIANGLES, 0, 3);
        f->glBindVertexArray(0);
        m->packing_program->release();

        if (blend_enabled)
            f->glEnable(GL_BLEND);
        if (cull_enabled)
            f->glEnable(GL_CULL_FACE);
        if (depth_enabled)
            f->glEnable(GL_DEPTH_TEST);
        if (scissor_enabled)
            f->glEnable(GL_SCISSOR_TEST);
        f->glColorMask(previous_colour_mask[0], previous_colour_mask[1], previous_colour_mask[2], previous_colour_mask[3]);
        f->glViewport(previous_viewport[0], previous_viewport[1], previous_viewport[2], previous_viewport[3]);
        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, GLuint(previous_draw_framebuffer));
    }

    {
        GLint previous_read_framebuffer = 0;
        GLint previous_read_buffer = 0;
        GLint previous_pack_alignment = 0;
        f->glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previous_read_framebuffer);
        f->glGetIntegerv(GL_READ_BUFFER, &previous_read_buffer);
        f->glGetIntegerv(GL_PACK_ALIGNMENT, &previous_pack_alignment);
        f->glBindFramebuffer(GL_READ_FRAMEBUFFER, m->packing_framebuffer);
        f->glReadBuffer(GL_COLOR_ATTACHMENT0);
        f->glPixelStorei(GL_PACK_ALIGNMENT, 1);
        f->glBindBuffer(GL_PIXEL_PACK_BUFFER, m->encoded_buffer);
        f->glReadPixels(0, 0, m->output_atlas_width, m->output_atlas_height, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, nullptr);
        f->glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        f->glPixelStorei(GL_PACK_ALIGNMENT, previous_pack_alignment);
        f->glBindFramebuffer(GL_READ_FRAMEBUFFER, GLuint(previous_read_framebuffer));
        f->glReadBuffer(GLenum(previous_read_buffer));
    }

    {
        f->glBindTexture(GL_TEXTURE_2D_ARRAY, destination.m_id);
        f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m->encoded_buffer);
        const auto format = Texture::compressed_texture_format();
        for (unsigned level = 0; level < result.mip_levels; ++level) {
            const auto level_width = std::max(1u, m->width >> level);
            const auto level_height = std::max(1u, m->height >> level);
            const auto layer_size = compressed_level_size(level_width, level_height);
            for (size_t layer = 0; layer < textures.size(); ++layer) {
                const auto offset = level_offsets[level] + layer_size * layer;
                f->glCompressedTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                    GLint(level),
                    0,
                    0,
                    GLint(destination_layers[layer]),
                    GLsizei(level_width),
                    GLsizei(level_height),
                    1,
                    format,
                    GLsizei(layer_size),
                    reinterpret_cast<const void*>(quintptr(offset)));
            }
        }
        f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    f->glActiveTexture(GL_TEXTURE0);
    return result;
}
