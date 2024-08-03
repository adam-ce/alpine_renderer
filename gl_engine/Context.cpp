/*****************************************************************************
 * Alpine Terrain Renderer
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

#include "Context.h"
#include "ShaderManager.h"
#include "TrackManager.h"
using namespace gl_engine;

Context::Context()
{
    m_shader_manager = std::make_unique<ShaderManager>();
    m_track_manager = std::make_unique<TrackManager>(m_shader_manager.get());
}

Context::~Context() = default;

void Context::setup_tracks(nucleus::track::Manager* manager)
{
    connect(manager, &nucleus::track::Manager::tracks_changed, m_track_manager.get(), &TrackManager::change_tracks);
}

void Context::deinit()
{
    m_track_manager.reset();
    m_shader_manager.reset();
}

TrackManager* Context::track_manager() { return m_track_manager.get(); }

ShaderManager* Context::shader_manager() const { return m_shader_manager.get(); }
