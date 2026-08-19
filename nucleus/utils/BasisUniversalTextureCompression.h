/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#pragma once

#include "ColourTexture.h"

#include <cstddef>
#include <expected>
#include <string>

namespace nucleus::utils {

enum class BasisUniversalFormat { ETC1S, UASTC_LDR_4x4, XUASTC_LDR_4x4 };

struct BasisUniversalCompressionSettings {
    BasisUniversalFormat format = BasisUniversalFormat::ETC1S;
    ColourTexture::Format target_format = ColourTexture::Format::DXT1;
    int quality = 75;
    int effort = 4;
    bool generate_mipmaps = true;
};

struct BasisUniversalCompressionTimings {
    double source_preparation_ms = 0.0;
    double encoding_ms = 0.0;
    double transcoding_ms = 0.0;

    [[nodiscard]] double total_ms() const { return source_preparation_ms + encoding_ms + transcoding_ms; }
};

struct BasisUniversalCompressionResult {
    MipmappedColourTexture texture;
    BasisUniversalCompressionTimings timings;
    size_t intermediate_bytes = 0;
    size_t transcoded_bytes = 0;
};

[[nodiscard]] const char* basis_universal_format_name(BasisUniversalFormat format);
[[nodiscard]] std::expected<BasisUniversalCompressionResult, std::string> compress_with_basis_universal(
    const radix::Raster<glm::u8vec4>& source, const BasisUniversalCompressionSettings& settings);

} // namespace nucleus::utils
