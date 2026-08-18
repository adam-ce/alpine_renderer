uniform highp usampler2D encoded_blocks;
uniform highp int block_atlas_width;
uniform highp int output_atlas_width;
uniform highp int total_blocks;
layout(location = 0) out highp uvec4 encoded_pixel;

void main()
{
    highp int output_pixel = int(gl_FragCoord.y) * output_atlas_width + int(gl_FragCoord.x);
    if (output_pixel >= total_blocks * 2)
        discard;

    highp int block_index = output_pixel / 2;
    highp ivec2 block_position = ivec2(block_index % block_atlas_width, block_index / block_atlas_width);
    highp uvec2 encoded = texelFetch(encoded_blocks, block_position, 0).rg;
    highp uint word = output_pixel % 2 == 0 ? encoded.x : encoded.y;
    encoded_pixel = uvec4(word & 0xffu, (word >> 8u) & 0xffu, (word >> 16u) & 0xffu, word >> 24u);
}
