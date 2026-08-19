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

#pragma once

#include <QImage>
#include <cstdint>
#include <memory>
#include <optional>
#include <qopengl.h>
#include <span>
#include <vector>
#ifdef ANDROID
#include <GLES3/gl3.h>
#endif
#include <radix/raster.h>
#include <nucleus/utils/ColourTexture.h>

namespace gl_engine {
class TextureCompressor;

class Texture {
public:
    enum class Target : GLenum { _2d = GL_TEXTURE_2D, _2dArray = GL_TEXTURE_2D_ARRAY }; // no 1D textures in webgl
    enum class Format {
        RGBA8, // normalised on gpu
        SRGBA8, // normalised on gpu
        CompressedRGBA8, // normalised on gpu, compression format depends on desktop/mobile
        RGBA8UI,
        RGBA32F,
        RG8, // normalised on gpu
        RG32UI,
        RGB32UI,
        R8UI,
        R16UI,
        R32UI,
        Invalid
    };
    enum class Filter : GLint { Nearest = GL_NEAREST, Linear = GL_LINEAR, MipMapLinear = GL_LINEAR_MIPMAP_LINEAR };

public:
    Texture(const Texture&) = delete;
    Texture(Texture&&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&&) = delete;
    explicit Texture(Target target, Format format);
    ~Texture();

    void bind(unsigned texture_unit);
    void setParams(Filter min_filter, Filter mag_filter, bool anisotropic_filtering = false);
    void allocate_array(unsigned width, unsigned height, unsigned n_layers);
    void upload(const nucleus::utils::ColourTexture& texture);
    void upload(const nucleus::utils::ColourTexture& texture, unsigned array_index);
    void upload(const nucleus::utils::MipmappedColourTexture& mipped_texture, unsigned array_index);
    template <typename T> void upload(const radix::Raster<T>& texture, unsigned int array_index);
    template <typename T> void upload(const radix::Raster<T>& texture);

    static GLenum compressed_texture_format();
    static nucleus::utils::ColourTexture::Format compression_algorithm();

protected:
    static GLenum max_anisotropy_param();
    static float max_anisotropy();

private:
    friend class TextureCompressor;

    GLuint m_id = GLuint(-1);
    Target m_target = Target::_2d;
    Format m_format = Format::Invalid;
    Filter m_min_filter = Filter::Nearest;
    Filter m_mag_filter = Filter::Nearest;
    unsigned m_width = unsigned(-1);
    unsigned m_height = unsigned(-1);
    unsigned m_n_layers = unsigned(-1);
};

class TextureCompressor {
public:
    enum class TimingMode { EndToEnd, IndividualStages, SubmissionOnly };

    struct GpuTimings {
        double scratch_upload_ms = 0.0;
        double mipmap_generation_ms = 0.0;
        double compression_pass_ms = 0.0;
        double packing_pass_ms = 0.0;
        double output_transfer_ms = 0.0;
        double compressed_upload_ms = 0.0;

        [[nodiscard]] double total_ms() const;
    };

    class GpuTimer {
    public:
        enum class PollStatus { Pending, Ready, Disjoint };

        GpuTimer();
        ~GpuTimer();
        GpuTimer(const GpuTimer&) = delete;
        GpuTimer(GpuTimer&&) = delete;
        GpuTimer& operator=(const GpuTimer&) = delete;
        GpuTimer& operator=(GpuTimer&&) = delete;

        [[nodiscard]] bool is_supported() const;
        [[nodiscard]] PollStatus poll(uint64_t ticket, GpuTimings& timings);

    private:
        friend class TextureCompressor;
        enum class Stage { ScratchUpload, MipmapGeneration, CompressionPass, PackingPass, OutputTransfer, CompressedUpload };

        [[nodiscard]] uint64_t begin_sample();
        void begin_stage(Stage stage);
        void end_stage();
        void end_sample();

        struct Impl;
        std::unique_ptr<Impl> m;
    };

    struct Settings {
        nucleus::utils::ColourTexture::Format algorithm = nucleus::utils::ColourTexture::Format::DXT1;
        unsigned effort = 0;
        bool generate_mipmaps = true;
        TimingMode timing_mode = TimingMode::EndToEnd;
        GpuTimer* gpu_timer = nullptr;
    };

    struct StageTiming {
        double submission_ms = 0.0;
        double completion_wait_ms = 0.0;

        [[nodiscard]] double total_ms() const { return submission_ms + completion_wait_ms; }
    };

    struct Timings {
        // EndToEnd includes one final completion wait. IndividualStages includes a completion
        // wait per stage. SubmissionOnly does not wait for GPU completion.
        StageTiming scratch_upload;
        StageTiming mipmap_generation;
        StageTiming compression_pass;
        StageTiming packing_pass;
        StageTiming encoding;
        StageTiming output_transfer;
        StageTiming compressed_upload;
        double total_ms = 0.0;
    };

    struct Result {
        Timings timings;
        size_t encoded_bytes = 0;
        unsigned mip_levels = 0;
        uint64_t gpu_timing_ticket = 0;
    };

    TextureCompressor(unsigned width, unsigned height, unsigned max_batch_size);
    ~TextureCompressor();
    TextureCompressor(const TextureCompressor&) = delete;
    TextureCompressor(TextureCompressor&&) = delete;
    TextureCompressor& operator=(const TextureCompressor&) = delete;
    TextureCompressor& operator=(TextureCompressor&&) = delete;

    [[nodiscard]] Result compress(std::span<const radix::Raster<glm::u8vec4>> textures,
        Texture& destination,
        std::span<const unsigned> destination_layers,
        const Settings& settings);

    [[nodiscard]] static size_t compressed_level_size(unsigned width, unsigned height);
    [[nodiscard]] static unsigned mip_level_count(unsigned width, unsigned height);
    [[nodiscard]] static bool is_supported();

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

extern template void gl_engine::Texture::upload<uint16_t>(const radix::Raster<uint16_t>&);
extern template void gl_engine::Texture::upload<uint32_t>(const radix::Raster<uint32_t>&);
extern template void gl_engine::Texture::upload<glm::vec<2, uint32_t>>(const radix::Raster<glm::vec<2, uint32_t>>&);
extern template void gl_engine::Texture::upload<glm::vec<3, uint32_t>>(const radix::Raster<glm::vec<3, uint32_t>>&);
extern template void gl_engine::Texture::upload<glm::vec<2, uint8_t>>(const radix::Raster<glm::vec<2, uint8_t>>&);
extern template void gl_engine::Texture::upload<glm::vec<4, uint8_t>>(const radix::Raster<glm::vec<4, uint8_t>>&);

extern template void gl_engine::Texture::upload<uint32_t>(const radix::Raster<uint32_t>&, unsigned int);
extern template void gl_engine::Texture::upload<glm::vec<2, uint32_t>>(const radix::Raster<glm::vec<2, uint32_t>>&, unsigned int);
extern template void gl_engine::Texture::upload<glm::vec<3, uint32_t>>(const radix::Raster<glm::vec<3, uint32_t>>&, unsigned int);

} // namespace gl_engine
