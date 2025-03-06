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

#include "TextureLayer.h"

#include "ShaderProgram.h"
#include "ShaderRegistry.h"
#include "Texture.h"
#include "TileGeometry.h"
#include <QOpenGLExtraFunctions>

namespace gl_engine {

TextureLayer::TextureLayer(unsigned int resolution, QObject* parent)
    : QObject { parent }
    , m_resolution(resolution)
{
}

void gl_engine::TextureLayer::init(ShaderRegistry* shader_registry)
{
    m_shader = std::make_shared<ShaderProgram>("tile.vert", "tile.frag");
    shader_registry->add_shader(m_shader);

    m_ortho_textures = std::make_unique<Texture>(Texture::Target::_2dArray, Texture::Format::CompressedRGBA8);
    m_ortho_textures->setParams(Texture::Filter::MipMapLinear, Texture::Filter::Linear, true);
    m_ortho_textures->allocate_array(m_resolution, m_resolution, unsigned(m_gpu_array_helper.size()));

    m_tile_id_texture = std::make_unique<Texture>(Texture::Target::_2d, Texture::Format::RG32UI);
    m_tile_id_texture->setParams(Texture::Filter::Nearest, Texture::Filter::Nearest);

    m_array_index_texture = std::make_unique<Texture>(Texture::Target::_2d, Texture::Format::R16UI);
    m_array_index_texture->setParams(Texture::Filter::Nearest, Texture::Filter::Nearest);

    m_instance_zoom_texture = std::make_unique<Texture>(Texture::Target::_2d, Texture::Format::R8UI);
    m_instance_zoom_texture->setParams(Texture::Filter::Nearest, Texture::Filter::Nearest);

    m_instance_array_index_texture = std::make_unique<Texture>(Texture::Target::_2d, Texture::Format::R16UI);
    m_instance_array_index_texture->setParams(Texture::Filter::Nearest, Texture::Filter::Nearest);

    m_zoom_level_ubo = std::make_unique<UniformBuffer<std::array<uint8_t, 1024>>>(0, "texture_layer_zoom_level");
    m_zoom_level_ubo->init();
    m_zoom_level_ubo->bind_to_shader(m_shader.get());

    m_array_index_ubo = std::make_unique<UniformBuffer<std::array<unsigned short, 1024>>>(1, "texture_layer_array_index");
    m_array_index_ubo->init();
    m_array_index_ubo->bind_to_shader(m_shader.get());

    update_gpu_id_map();
}

void TextureLayer::draw(const TileGeometry& tile_geometry,
    const nucleus::camera::Definition& camera,
    const nucleus::tile::DrawListGenerator::TileSet& draw_tiles,
    bool sort_tiles,
    glm::dvec3 sort_position) const
{
    m_shader->bind();
    m_shader->set_uniform("ortho_sampler", 2);
    m_ortho_textures->bind(2);

    m_shader->set_uniform("ortho_map_index_sampler", 5);
    m_array_index_texture->bind(5);
    m_shader->set_uniform("ortho_map_tile_id_sampler", 6);
    m_tile_id_texture->bind(6);

    const auto draw_list = tile_geometry.sort(camera, draw_tiles);
    nucleus::Raster<uint8_t> zoom_level_raster = { glm::uvec2 { 1024, 1 } };
    nucleus::Raster<uint16_t> array_index_raster = { glm::uvec2 { 1024, 1 } };
    for (unsigned i = 0; i < std::min(unsigned(draw_list.size()), 1024u); ++i) {
        const auto layer = m_gpu_array_helper.layer(draw_list[i]);
        zoom_level_raster.pixel({ i, 0 }) = layer.id.zoom_level;
        array_index_raster.pixel({ i, 0 }) = layer.index;
    }

    m_shader->set_uniform("ortho_map_instance_index_sampler", 7);
    m_instance_array_index_texture->bind(7);
    m_instance_array_index_texture->upload(array_index_raster);
    m_instance_zoom_texture->bind(8);
    m_shader->set_uniform("ortho_map_instance_zoom_sampler", 8);
    m_instance_zoom_texture->upload(zoom_level_raster);

    // std::array<uint8_t, 1024> zoom_level_arr = {};
    // std::array<uint16_t, 1024> array_index_arr = {};
    // for (unsigned i = 0; i < std::min(unsigned(draw_list.size()), 1024u); ++i) {
    //     const auto layer = m_gpu_array_helper.layer(draw_list[i]);
    //     zoom_level_arr[i] = layer.id.zoom_level;
    //     array_index_arr[i] = layer.index;
    // }
    // m_zoom_level_ubo->data = zoom_level_arr;
    // m_zoom_level_ubo->update_gpu_data();
    // m_zoom_level_ubo->bind_to_shader(m_shader.get());
    // m_array_index_ubo->data = array_index_arr;
    // m_array_index_ubo->update_gpu_data();
    // m_array_index_ubo->bind_to_shader(m_shader.get());

    // todo:
    // use info.
    // delete dict access

    tile_geometry.draw(m_shader.get(), camera, draw_tiles, sort_tiles, sort_position);
}

unsigned TextureLayer::tile_count() const { return m_gpu_array_helper.n_occupied(); }

void TextureLayer::update_gpu_tiles(const std::vector<nucleus::tile::Id>& deleted_tiles, const std::vector<nucleus::tile::GpuTextureTile>& new_tiles)
{
    if (!QOpenGLContext::currentContext()) // can happen during shutdown.
        return;

    for (const auto& tile_id : deleted_tiles) {
        m_gpu_array_helper.remove_tile(tile_id);
    }
    for (const auto& tile : new_tiles) {
        // test for validity
        assert(tile.id.zoom_level < 100);
        assert(tile.texture);

        // find empty spot and upload texture
        const auto layer_index = m_gpu_array_helper.add_tile(tile.id);
        m_ortho_textures->upload(*tile.texture, layer_index);
    }
    update_gpu_id_map();
}

void TextureLayer::set_quad_limit(unsigned int new_limit) { m_gpu_array_helper.set_quad_limit(new_limit); }

void TextureLayer::update_gpu_id_map()
{
    auto [packed_ids, layers] = m_gpu_array_helper.generate_dictionary();
    m_array_index_texture->upload(layers);
    m_tile_id_texture->upload(packed_ids);
}

} // namespace gl_engine
