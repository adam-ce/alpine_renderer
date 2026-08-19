/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#include "BasisUniversalTextureCompression.h"

#include <encoder/basisu_comp.h>
#include <transcoder/basisu_transcoder.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

basist::basis_tex_format basis_format(nucleus::utils::BasisUniversalFormat format)
{
    using Format = nucleus::utils::BasisUniversalFormat;
    switch (format) {
    case Format::ETC1S:
        return basist::basis_tex_format::cETC1S;
    case Format::UASTC_LDR_4x4:
        return basist::basis_tex_format::cUASTC_LDR_4x4;
    case Format::XUASTC_LDR_4x4:
        return basist::basis_tex_format::cXUASTC_LDR_4x4;
    }
    return basist::basis_tex_format::cETC1S;
}

basist::transcoder_texture_format transcoder_format(nucleus::utils::ColourTexture::Format format)
{
    using Format = nucleus::utils::ColourTexture::Format;
    switch (format) {
    case Format::DXT1:
        return basist::transcoder_texture_format::cTFBC1_RGB;
    case Format::ETC1:
        return basist::transcoder_texture_format::cTFETC1_RGB;
    case Format::Uncompressed_RGBA:
        break;
    }
    return basist::transcoder_texture_format::cTFRGBA32;
}

std::vector<radix::Raster<glm::u8vec4>> mip_levels(
    const radix::Raster<glm::u8vec4>& source, bool generate_mipmaps)
{
    if (!generate_mipmaps)
        return { source };
    auto levels = radix::raster::generate_mipmap(source);
    return levels ? std::move(*levels) : std::vector<radix::Raster<glm::u8vec4>> {};
}

basisu::vector<basisu::image> basis_images(const std::vector<radix::Raster<glm::u8vec4>>& levels)
{
    static_assert(sizeof(basisu::color_rgba) == sizeof(glm::u8vec4));
    basisu::vector<basisu::image> result;
    result.reserve(levels.size());
    for (const auto& level : levels) {
        basisu::image image(level.width(), level.height());
        std::memcpy(image.get_ptr(), level.bytes().data(), level.size_in_bytes());
        result.push_back(std::move(image));
    }
    return result;
}

void initialise_basis_universal()
{
    static std::once_flag flag;
    std::call_once(flag, [] { basisu::basisu_encoder_init(); });
}

} // namespace

const char* nucleus::utils::basis_universal_format_name(BasisUniversalFormat format)
{
    switch (format) {
    case BasisUniversalFormat::ETC1S:
        return "BasisU ETC1S";
    case BasisUniversalFormat::UASTC_LDR_4x4:
        return "BasisU UASTC LDR 4x4";
    case BasisUniversalFormat::XUASTC_LDR_4x4:
        return "BasisU XUASTC LDR 4x4";
    }
    return "BasisU unknown";
}

std::expected<nucleus::utils::BasisUniversalCompressionResult, std::string>
nucleus::utils::compress_with_basis_universal(
    const radix::Raster<glm::u8vec4>& source, const BasisUniversalCompressionSettings& settings)
{
    if (source.buffer().empty())
        return std::unexpected("Cannot compress an empty image");
    if (settings.target_format == ColourTexture::Format::Uncompressed_RGBA)
        return std::unexpected("Basis Universal target must be BC1 or ETC1");
    if (source.width() > int(basist::BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION)
        || source.height() > int(basist::BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION)) {
        return std::unexpected("Image exceeds Basis Universal's maximum dimensions");
    }

    initialise_basis_universal();
    BasisUniversalCompressionResult result;

    const auto preparation_start = Clock::now();
    const auto levels = mip_levels(source, settings.generate_mipmaps);
    if (levels.empty())
        return std::unexpected("Basis Universal requires power-of-two mipmap input");
    auto images = basis_images(levels);
    result.timings.source_preparation_ms = elapsed_ms(preparation_start);

    const auto encoding_start = Clock::now();
    size_t encoded_size = 0;
    const auto flags = uint32_t(basisu::cFlagSRGB);
    void* encoded = basisu::basis_compress2(basis_format(settings.format),
        images,
        flags,
        std::clamp(settings.quality, 1, 100),
        std::clamp(settings.effort, 0, 10),
        &encoded_size);
    result.timings.encoding_ms = elapsed_ms(encoding_start);
    if (!encoded)
        return std::unexpected("Basis Universal encoding failed");
    const auto encoded_deleter = [](void* data) { basisu::basis_free_data(data); };
    std::unique_ptr<void, decltype(encoded_deleter)> encoded_owner(encoded, encoded_deleter);
    result.intermediate_bytes = encoded_size;
    if (encoded_size > std::numeric_limits<uint32_t>::max())
        return std::unexpected("Basis Universal output is too large to transcode");

    const auto transcoding_start = Clock::now();
    basist::basisu_transcoder transcoder;
    const auto encoded_size_u32 = uint32_t(encoded_size);
    if (!transcoder.validate_header(encoded, encoded_size_u32))
        return std::unexpected("Basis Universal produced an invalid header");
    if (transcoder.get_total_images(encoded, encoded_size_u32) != 1)
        return std::unexpected("Basis Universal produced an unexpected image count");
    if (!transcoder.start_transcoding(encoded, encoded_size_u32))
        return std::unexpected("Basis Universal transcoder initialisation failed");

    const auto level_count = transcoder.get_total_image_levels(encoded, encoded_size_u32, 0);
    if (level_count != levels.size())
        return std::unexpected("Basis Universal produced an unexpected mip level count");
    result.texture.reserve(level_count);
    for (uint32_t level_index = 0; level_index < level_count; ++level_index) {
        basist::basisu_image_level_info info;
        if (!transcoder.get_image_level_info(encoded, encoded_size_u32, info, 0, level_index))
            return std::unexpected("Unable to inspect a Basis Universal mip level");
        const auto block_count = ((info.m_orig_width + 3u) / 4u) * ((info.m_orig_height + 3u) / 4u);
        std::vector<uint8_t> blocks(size_t(block_count) * 8u);
        if (!transcoder.transcode_image_level(encoded,
                encoded_size_u32,
                0,
                level_index,
                blocks.data(),
                block_count,
                transcoder_format(settings.target_format))) {
            return std::unexpected("Basis Universal transcoding failed");
        }
        result.transcoded_bytes += blocks.size();
        result.texture.emplace_back(std::move(blocks), info.m_orig_width, info.m_orig_height, settings.target_format);
    }
    result.timings.transcoding_ms = elapsed_ms(transcoding_start);
    return result;
}
