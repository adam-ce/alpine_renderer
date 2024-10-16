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

#include "Scheduler.h"
#include <QThread>
#include <memory>
#include <nucleus/tile_scheduler/QuadAssembler.h>
#include <nucleus/tile_scheduler/RateLimiter.h>
#include <nucleus/tile_scheduler/SlotLimiter.h>
#include <nucleus/tile_scheduler/TileLoadService.h>

namespace nucleus::map_label::setup {

using TileLoadServicePtr = std::unique_ptr<nucleus::tile_scheduler::TileLoadService>;

struct SchedulerHolder {
    std::unique_ptr<map_label::Scheduler> scheduler;
    TileLoadServicePtr vector_service;
};

SchedulerHolder scheduler(TileLoadServicePtr vector_service,
    const tile_scheduler::utils::AabbDecoratorPtr& aabb_decorator,
    const std::shared_ptr<nucleus::DataQuerier>& data_querier,
    QThread* thread = nullptr)
{
    auto scheduler = std::make_unique<nucleus::map_label::Scheduler>();
    scheduler->read_disk_cache();
    scheduler->set_gpu_quad_limit(512);
    scheduler->set_ram_quad_limit(12000);
    scheduler->set_aabb_decorator(aabb_decorator);
    scheduler->set_dataquerier(data_querier);

    {
        using nucleus::tile_scheduler::QuadAssembler;
        using nucleus::tile_scheduler::RateLimiter;
        using nucleus::tile_scheduler::SlotLimiter;
        using nucleus::tile_scheduler::TileLoadService;
        auto* sch = scheduler.get();
        auto* sl = new SlotLimiter(sch);
        auto* rl = new RateLimiter(sch);
        auto* qa = new QuadAssembler(sch);

        QObject::connect(sch, &Scheduler::quads_requested, sl, &SlotLimiter::request_quads);
        QObject::connect(sl, &SlotLimiter::quad_requested, rl, &RateLimiter::request_quad);
        QObject::connect(rl, &RateLimiter::quad_requested, qa, &QuadAssembler::load);
        QObject::connect(qa, &QuadAssembler::tile_requested, vector_service.get(), &TileLoadService::load);
        QObject::connect(vector_service.get(), &TileLoadService::load_finished, qa, &QuadAssembler::deliver_tile);

        QObject::connect(qa, &QuadAssembler::quad_loaded, sl, &SlotLimiter::deliver_quad);
        QObject::connect(sl, &SlotLimiter::quad_delivered, sch, &nucleus::map_label::Scheduler::receive_quad);
    }
    if (QNetworkInformation::loadDefaultBackend() && QNetworkInformation::instance()) {
        QNetworkInformation* n = QNetworkInformation::instance();
        scheduler->set_network_reachability(n->reachability());
        QObject::connect(n, &QNetworkInformation::reachabilityChanged, scheduler.get(), &Scheduler::set_network_reachability);
    }

#ifdef ALP_ENABLE_THREADING
#ifdef __EMSCRIPTEN__ // make request from main thread on webassembly due to QTBUG-109396
    m_vectortile_service->moveToThread(QCoreApplication::instance()->thread());
#else
    if (thread)
        vector_service->moveToThread(thread);
#endif
    if (thread)
        scheduler->moveToThread(thread);
#endif

    return { std::move(scheduler), std::move(vector_service) };
}
} // namespace nucleus::map_label::setup
