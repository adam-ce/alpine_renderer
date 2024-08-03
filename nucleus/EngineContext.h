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

#pragma once

#include <QObject>

#include "AbstractRenderWindow.h"
#include "nucleus/track/Manager.h"

namespace nucleus {

class EngineContext : public QObject {
    Q_OBJECT
public:
    explicit EngineContext(QObject* parent = nullptr);

    virtual std::weak_ptr<AbstractRenderWindow> render_window() = 0;
    virtual void setup_tracks(track::Manager* manager) = 0;
signals:
};
} // namespace nucleus
