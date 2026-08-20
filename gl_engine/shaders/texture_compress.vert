uniform highp sampler2DArray source_texture;
uniform highp int texture_width;
uniform highp int texture_height;
const highp int max_mip_levels = 16;
uniform highp int atlas_width;
uniform highp int total_blocks;
uniform highp int mip_levels;
uniform highp int level_offsets[max_mip_levels];
uniform highp int level_blocks_x[max_mip_levels];
uniform highp int level_blocks_y[max_mip_levels];
layout(location = 0) out highp uvec2 encoded_block;
uniform highp int effort;

highp uvec3 unpack_565(highp uint value)
{
    return uvec3(((value >> 11u) & 31u) * 255u / 31u,
        ((value >> 5u) & 63u) * 255u / 63u,
        (value & 31u) * 255u / 31u);
}

highp uint pack_565(highp uvec3 value)
{
    return ((value.r * 31u + 127u) / 255u) << 11u | ((value.g * 63u + 127u) / 255u) << 5u | (value.b * 31u + 127u) / 255u;
}

highp uint colour_error(highp uvec3 lhs, highp uvec3 rhs)
{
    highp ivec3 delta = ivec3(lhs) - ivec3(rhs);
    return uint(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
}

highp uvec2 encode_dxt1(highp uvec3 pixels[16])
{
    highp uvec3 minimum_colour = uvec3(255u);
    highp uvec3 maximum_colour = uvec3(0u);
    for (int i = 0; i < 16; ++i) {
        minimum_colour = min(minimum_colour, pixels[i]);
        maximum_colour = max(maximum_colour, pixels[i]);
    }

    highp uint best_error = 0xffffffffu;
    highp uint best_endpoints = 0u;
    highp uint best_indices = 0u;
    for (int candidate = 0; candidate <= 10; ++candidate) {
        if (candidate > effort)
            break;
        highp uvec3 range = maximum_colour - minimum_colour;
        highp uvec3 inset = range * uint(candidate) / 64u;
        highp uint colour0 = pack_565(maximum_colour - inset);
        highp uint colour1 = pack_565(minimum_colour + inset);
        if (colour0 <= colour1) {
            highp uint swap_value = colour0;
            colour0 = colour1;
            colour1 = swap_value;
        }
        if (colour0 == colour1) {
            if (colour0 < 65535u)
                ++colour0;
            else
                --colour1;
        }

        highp uvec3 palette[4];
        palette[0] = unpack_565(colour0);
        palette[1] = unpack_565(colour1);
        palette[2] = (2u * palette[0] + palette[1]) / 3u;
        palette[3] = (palette[0] + 2u * palette[1]) / 3u;

        highp uint total_error = 0u;
        highp uint indices = 0u;
        for (int i = 0; i < 16; ++i) {
            highp uint selected = 0u;
            highp uint selected_error = colour_error(pixels[i], palette[0]);
            for (uint palette_index = 1u; palette_index < 4u; ++palette_index) {
                highp uint error = colour_error(pixels[i], palette[palette_index]);
                if (error < selected_error) {
                    selected = palette_index;
                    selected_error = error;
                }
            }
            total_error += selected_error;
            indices |= selected << uint(2 * i);
        }
        if (total_error < best_error) {
            best_error = total_error;
            best_endpoints = colour0 | colour1 << 16u;
            best_indices = indices;
        }
    }
    return uvec2(best_endpoints, best_indices);
}

highp uint byte_swap(highp uint value)
{
    return value >> 24u | (value >> 8u & 0x0000ff00u) | (value << 8u & 0x00ff0000u) | value << 24u;
}

highp int modifier(highp int table, highp int index)
{
    const highp ivec4 modifiers[8] = ivec4[8](ivec4(2, 8, -2, -8),
        ivec4(5, 17, -5, -17),
        ivec4(9, 29, -9, -29),
        ivec4(13, 42, -13, -42),
        ivec4(18, 60, -18, -60),
        ivec4(24, 80, -24, -80),
        ivec4(33, 106, -33, -106),
        ivec4(47, 183, -47, -183));
    return modifiers[table][index];
}

highp int brightness(highp uvec3 colour)
{
    return int((colour.r + 2u * colour.g + colour.b + 2u) / 4u);
}

highp int table_for_range(highp int range)
{
    if (range < 22)
        return 0;
    if (range < 44)
        return 1;
    if (range < 74)
        return 2;
    if (range < 106)
        return 3;
    if (range < 152)
        return 4;
    if (range < 182)
        return 5;
    if (range < 254)
        return 6;
    return 7;
}

highp uvec2 encode_etc1_fast(highp uvec3 pixels[16])
{
    highp uvec3 minimum_colour = uvec3(255u);
    highp uvec3 maximum_colour = uvec3(0u);
    highp uvec3 sum = uvec3(0u);
    for (int i = 0; i < 16; ++i) {
        minimum_colour = min(minimum_colour, pixels[i]);
        maximum_colour = max(maximum_colour, pixels[i]);
        sum += pixels[i];
    }

    highp int minimum_brightness = brightness(minimum_colour);
    highp int maximum_brightness = brightness(maximum_colour);
    highp int range = max(8, maximum_brightness - minimum_brightness);
    highp int middle = (minimum_brightness + maximum_brightness + 1) / 2;
    highp ivec3 average = ivec3((sum + 8u) / 16u);
    highp int correction = middle - brightness(uvec3(average));
    highp ivec3 adjusted = clamp(average + ivec3(correction), ivec3(0), ivec3(255));
    highp uvec3 base5 = (uvec3(adjusted) * 31u + 127u) / 255u;

    highp int table = table_for_range(range);
    highp int threshold = (range * 3 + 4) / 8;
    highp uint indices = 0u;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            highp int pixel_index = y * 4 + x;
            highp int delta = brightness(pixels[pixel_index]) - middle;
            highp uint selected = uint(abs(delta) >= threshold);
            if (delta < 0)
                selected += 2u;
            highp uint bit_position = uint(x * 4 + y);
            indices |= (selected & 1u) << bit_position;
            indices |= (selected >> 1u) << (bit_position + 16u);
        }
    }

    highp uint control = uint(table) << 5u | uint(table) << 2u | 2u;
    highp uint header = base5.r << 3u | base5.g << 11u | base5.b << 19u | control << 24u;
    return uvec2(header, byte_swap(indices));
}

struct FastEtc1Subblock {
    highp ivec3 colour;
    highp int table;
    highp int middle;
    highp int threshold;
};

struct FastEtc1Block {
    highp uvec2 encoded;
    highp uint error;
};

FastEtc1Subblock fast_subblock_parameters(highp uvec3 pixels[16], bool flip, highp int subblock)
{
    highp uvec3 minimum_colour = uvec3(255u);
    highp uvec3 maximum_colour = uvec3(0u);
    highp uvec3 sum = uvec3(0u);
    for (int i = 0; i < 16; ++i) {
        highp int x = i & 3;
        highp int y = i >> 2;
        bool belongs_to_first = flip ? y < 2 : x < 2;
        if (belongs_to_first != (subblock == 0))
            continue;
        minimum_colour = min(minimum_colour, pixels[i]);
        maximum_colour = max(maximum_colour, pixels[i]);
        sum += pixels[i];
    }

    highp int minimum_brightness = brightness(minimum_colour);
    highp int maximum_brightness = brightness(maximum_colour);
    highp int range = max(8, maximum_brightness - minimum_brightness);
    highp int middle = (minimum_brightness + maximum_brightness + 1) / 2;
    highp ivec3 average = ivec3((sum + 4u) / 8u);
    highp int correction = middle - brightness(uvec3(average));
    highp ivec3 colour = clamp(average + ivec3(correction), ivec3(0), ivec3(255));
    return FastEtc1Subblock(colour, table_for_range(range), middle, (range * 3 + 4) / 8);
}

FastEtc1Block encode_etc1_split_orientation(highp uvec3 pixels[16], bool flip)
{
    FastEtc1Subblock first = fast_subblock_parameters(pixels, flip, 0);
    FastEtc1Subblock second = fast_subblock_parameters(pixels, flip, 1);
    highp uvec3 first_base5 = (uvec3(first.colour) * 31u + 127u) / 255u;
    highp uvec3 second_base5 = (uvec3(second.colour) * 31u + 127u) / 255u;
    highp ivec3 base_delta = ivec3(second_base5) - ivec3(first_base5);
    bool differential = all(greaterThanEqual(base_delta, ivec3(-4))) && all(lessThanEqual(base_delta, ivec3(3)));

    highp uint header;
    highp ivec3 first_decoded;
    highp ivec3 second_decoded;
    if (differential) {
        highp uvec3 delta3 = uvec3(base_delta) & 7u;
        header = first_base5.r << 3u | delta3.r
            | first_base5.g << 11u | delta3.g << 8u
            | first_base5.b << 19u | delta3.b << 16u;
        first_decoded = ivec3((first_base5 << 3u) | (first_base5 >> 2u));
        second_decoded = ivec3((second_base5 << 3u) | (second_base5 >> 2u));
    } else {
        highp uvec3 first_base4 = (uvec3(first.colour) * 15u + 127u) / 255u;
        highp uvec3 second_base4 = (uvec3(second.colour) * 15u + 127u) / 255u;
        header = first_base4.r << 4u | second_base4.r
            | first_base4.g << 12u | second_base4.g << 8u
            | first_base4.b << 20u | second_base4.b << 16u;
        first_decoded = ivec3((first_base4 << 4u) | first_base4);
        second_decoded = ivec3((second_base4 << 4u) | second_base4);
    }

    highp uint indices = 0u;
    highp uint total_error = 0u;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            highp int pixel_index = y * 4 + x;
            bool use_second = flip ? y >= 2 : x >= 2;
            highp int middle = use_second ? second.middle : first.middle;
            highp int threshold = use_second ? second.threshold : first.threshold;
            highp int table = use_second ? second.table : first.table;
            highp ivec3 decoded = use_second ? second_decoded : first_decoded;
            highp int brightness_delta = brightness(pixels[pixel_index]) - middle;
            highp uint selected = uint(abs(brightness_delta) >= threshold);
            if (brightness_delta < 0)
                selected += 2u;

            highp ivec3 reconstructed = clamp(decoded + ivec3(modifier(table, int(selected))), ivec3(0), ivec3(255));
            highp ivec3 colour_delta = ivec3(pixels[pixel_index]) - reconstructed;
            total_error += uint(colour_delta.x * colour_delta.x + colour_delta.y * colour_delta.y + colour_delta.z * colour_delta.z);
            highp uint bit_position = uint(x * 4 + y);
            indices |= (selected & 1u) << bit_position;
            indices |= (selected >> 1u) << (bit_position + 16u);
        }
    }

    highp uint control = uint(first.table) << 5u | uint(second.table) << 2u
        | (differential ? 2u : 0u) | (flip ? 1u : 0u);
    return FastEtc1Block(uvec2(header | control << 24u, byte_swap(indices)), total_error);
}

highp uvec2 encode_etc1_fast_split(highp uvec3 pixels[16])
{
    FastEtc1Block vertical = encode_etc1_split_orientation(pixels, false);
    FastEtc1Block horizontal = encode_etc1_split_orientation(pixels, true);
    return horizontal.error < vertical.error ? horizontal.encoded : vertical.encoded;
}

highp uvec2 encode_etc1(highp uvec3 pixels[16])
{
    highp uvec3 sum = uvec3(0u);
    for (int i = 0; i < 16; ++i)
        sum += pixels[i];
    highp ivec3 average = ivec3((sum + 8u) / 16u);

    highp uint best_error = 0xffffffffu;
    highp uvec3 best_base = uvec3(0u);
    highp uint best_table = 0u;
    highp uint best_indices = 0u;
    for (int candidate = 0; candidate <= 10; ++candidate) {
        if (candidate > effort)
            break;
        highp int magnitude = ((candidate + 1) / 2) * 4;
        highp int signed_offset = candidate == 0 ? 0 : ((candidate & 1) == 1 ? magnitude : -magnitude);
        highp ivec3 adjusted = clamp(average + ivec3(signed_offset), ivec3(0), ivec3(255));
        highp uvec3 base5 = (uvec3(adjusted) * 31u + 127u) / 255u;
        highp ivec3 decoded_base = ivec3((base5 << 3u) | (base5 >> 2u));

        for (int table = 0; table < 8; ++table) {
            highp uint total_error = 0u;
            highp uint indices = 0u;
            for (int y = 0; y < 4; ++y) {
                for (int x = 0; x < 4; ++x) {
                    highp int pixel_index = y * 4 + x;
                    highp uint selected = 0u;
                    highp uint selected_error = 0xffffffffu;
                    for (int index = 0; index < 4; ++index) {
                        highp ivec3 reconstructed = clamp(decoded_base + ivec3(modifier(table, index)), ivec3(0), ivec3(255));
                        highp ivec3 delta = ivec3(pixels[pixel_index]) - reconstructed;
                        highp uint error = uint(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
                        if (error < selected_error) {
                            selected = uint(index);
                            selected_error = error;
                        }
                    }
                    total_error += selected_error;
                    highp uint bit_position = uint(x * 4 + y);
                    indices |= (selected & 1u) << bit_position;
                    indices |= (selected >> 1u) << (bit_position + 16u);
                }
            }
            if (total_error < best_error) {
                best_error = total_error;
                best_base = base5;
                best_table = uint(table);
                best_indices = indices;
            }
        }
    }

    highp uint control = best_table << 5u | best_table << 2u | 2u;
    highp uint header = best_base.r << 3u | best_base.g << 11u | best_base.b << 19u | control << 24u;
    return uvec2(header, byte_swap(best_indices));
}

highp uvec2 compress_block(highp ivec2 block,
    highp int layer,
    highp int level,
    highp int level_width,
    highp int level_height)
{
    highp ivec2 origin = block * 4;
    highp uvec3 pixels[16];
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            highp ivec2 position = min(origin + ivec2(x, y), ivec2(level_width - 1, level_height - 1));
            pixels[y * 4 + x] = uvec3(round(texelFetch(source_texture, ivec3(position, layer), level).rgb * 255.0));
        }
    }

#ifdef ALP_COMPRESS_ETC1
#ifdef ALP_COMPRESS_ETC1_SPLIT
    return encode_etc1_fast_split(pixels);
#elif defined(ALP_COMPRESS_ETC1_FAST)
    return encode_etc1_fast(pixels);
#else
    return encode_etc1(pixels);
#endif
#else
    return encode_dxt1(pixels);
#endif
}

void main()
{
    highp int output_index = int(gl_FragCoord.y) * atlas_width + int(gl_FragCoord.x);
    if (output_index >= total_blocks)
        discard;

    highp int level = 0;
    for (int candidate = 1; candidate < max_mip_levels; ++candidate) {
        if (candidate >= mip_levels || output_index < level_offsets[candidate])
            break;
        level = candidate;
    }

    highp int blocks_x_at_level = level_blocks_x[level];
    highp int blocks_y_at_level = level_blocks_y[level];
    highp int blocks_per_layer = blocks_x_at_level * blocks_y_at_level;
    highp int level_index = output_index - level_offsets[level];
    highp int layer = level_index / blocks_per_layer;
    highp int block_index = level_index - layer * blocks_per_layer;
    highp ivec2 block = ivec2(block_index % blocks_x_at_level, block_index / blocks_x_at_level);
    encoded_block = compress_block(block, layer, level, max(1, texture_width >> level), max(1, texture_height >> level));
}
