/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#include <catch2/catch_test_macros.hpp>

#include <nucleus/utils/BasisUniversalTextureCompression.h>

#include <array>

TEST_CASE("nucleus/basis_universal_texture_compression: transcodes all LDR paths")
{
    using nucleus::utils::BasisUniversalCompressionSettings;
    using nucleus::utils::BasisUniversalFormat;
    using nucleus::utils::ColourTexture;

    radix::Raster<glm::u8vec4> source(glm::uvec2(16), glm::u8vec4(0, 0, 0, 255));
    for (unsigned y = 0; y < source.height(); ++y) {
        for (unsigned x = 0; x < source.width(); ++x)
            source.pixel({ x, y }) = glm::u8vec4(x * 16, y * 16, (x + y) * 8, 255);
    }

    constexpr std::array formats {
        BasisUniversalFormat::ETC1S,
        BasisUniversalFormat::UASTC_LDR_4x4,
        BasisUniversalFormat::XUASTC_LDR_4x4,
    };
    constexpr std::array targets { ColourTexture::Format::DXT1, ColourTexture::Format::ETC1 };
    for (const auto format : formats) {
        for (const auto target : targets) {
            INFO(nucleus::utils::basis_universal_format_name(format));
            const auto result = nucleus::utils::compress_with_basis_universal(source,
                BasisUniversalCompressionSettings {
                    .format = format,
                    .target_format = target,
                    .quality = 50,
                    .effort = 0,
                    .generate_mipmaps = true,
                });
            REQUIRE(result);
            CHECK(result->texture.size() == 5);
            CHECK(result->texture.front().width() == 16);
            CHECK(result->texture.front().height() == 16);
            CHECK(result->texture.front().format() == target);
            CHECK(result->intermediate_bytes > 0);
            CHECK(result->transcoded_bytes == 184);
        }
    }
}
