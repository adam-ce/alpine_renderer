/*****************************************************************************
 * Alpine Terrain Renderer
 * Copyright (C) 2026 Adam Celerek <family name at cg tuwien ac at>
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

#include "Window.h"

#include <QGuiApplication>
#include <QThread>
#include <QTimer>
#include <algorithm>
#include <chrono>
#include <nucleus/DataQuerier.h>
#include <nucleus/camera/Controller.h>
#include <nucleus/camera/PositionStorage.h>
#include <nucleus/tile/GeometryScheduler.h>
#include <nucleus/tile/SchedulerDirector.h>
#include <nucleus/tile/TileLoadService.h>
#include <nucleus/tile/setup.h>
#include <nucleus/utils/ColourTexture.h>
#include <nucleus/utils/thread.h>
#include <webgpu/engine/Context.h>
#include <webgpu/engine/tile_mesh/TileMeshRenderer.h>

using namespace std::chrono_literals;
using DataQuerier = nucleus::DataQuerier;
using Scheduler = nucleus::tile::Scheduler;
using GeometryScheduler = nucleus::tile::GeometryScheduler;
using TextureScheduler = nucleus::tile::TextureScheduler;
using TileLoadService = nucleus::tile::TileLoadService;
using CameraController = nucleus::camera::Controller;

int main(int argc, char** argv)
{
    QGuiApplication app(argc, argv);
    QCoreApplication::setOrganizationName("AlpineMaps.org");
    QCoreApplication::setApplicationName("PlainWgpuRenderer");

    using UrlPattern = TileLoadService::UrlPattern;
    auto terrain_service = std::make_unique<TileLoadService>("https://alpinemaps.cg.tuwien.ac.at/tiles/alpine_png/", UrlPattern::ZXY, ".png");
    auto ortho_service = std::make_unique<TileLoadService>("https://gataki.cg.tuwien.ac.at/raw/basemap/tiles/", UrlPattern::ZYX_yPointingSouth, ".jpeg");

    auto decorator = nucleus::tile::setup::aabb_decorator();
    QThread scheduler_thread;
    nucleus::tile::SchedulerDirector director;

    auto geometry_scheduler = nucleus::tile::setup::geometry_scheduler(std::move(terrain_service), decorator, &scheduler_thread);
    director.check_in("geometry", geometry_scheduler.scheduler);
    auto data_querier = std::make_shared<DataQuerier>(&geometry_scheduler.scheduler->ram_cache());

    auto ortho_scheduler = nucleus::tile::setup::texture_scheduler(
        std::move(ortho_service), decorator, &scheduler_thread, { .tile_resolution = 256, .max_zoom_level = 20, .gpu_quad_limit = 512 });
    director.check_in("ortho", ortho_scheduler.scheduler);

    auto context = std::make_shared<webgpu_engine::Context>();
    context->set_aabb_decorator(decorator);

    auto tile_mesh_renderer = std::make_shared<webgpu_engine::TileMeshRenderer>(65, 512);
    tile_mesh_renderer->set_tile_limit(512);
    context->set_tile_mesh_renderer(tile_mesh_renderer);

    Window window(context);
    CameraController camera_controller(nucleus::camera::PositionStorage::instance()->get("grossglockner"), &window, data_querier.get());

    QObject::connect(&camera_controller, &CameraController::definition_changed, &window, [&](const nucleus::camera::Definition&) {
        QTimer::singleShot(5ms, &camera_controller, &CameraController::advance_camera);
    });

    QObject::connect(&window, &Window::initialisation_started, context.get(), [&]() { context->initialise(); });
    QObject::connect(&window, &Window::initialisation_started, geometry_scheduler.scheduler.get(), [&]() { geometry_scheduler.scheduler->set_enabled(true); });
    QObject::connect(&window, &Window::initialisation_started, &window, [&ortho_scheduler]() {
        nucleus::utils::thread::async_call(ortho_scheduler.scheduler.get(), [&ortho_scheduler]() {
            ortho_scheduler.scheduler->set_texture_compression_algorithm(nucleus::utils::ColourTexture::Format::Uncompressed_RGBA);
            ortho_scheduler.scheduler->set_enabled(true);
        });
    });

    // clang-format off
    QObject::connect(&camera_controller, &CameraController::definition_changed, geometry_scheduler.scheduler.get(), &Scheduler::update_camera);
    QObject::connect(&camera_controller, &CameraController::definition_changed, ortho_scheduler.scheduler.get(),    &Scheduler::update_camera);
    QObject::connect(&camera_controller, &CameraController::definition_changed, &window,                            &Window::update_camera);
    QObject::connect(geometry_scheduler.scheduler.get(), &GeometryScheduler::gpu_tiles_updated, tile_mesh_renderer.get(), &webgpu_engine::TileMeshRenderer::update_gpu_tiles_height);
    QObject::connect(geometry_scheduler.scheduler.get(), &GeometryScheduler::gpu_tiles_updated, &window,                  &Window::request_render);
    QObject::connect(ortho_scheduler.scheduler.get(),    &TextureScheduler::gpu_tiles_updated,  tile_mesh_renderer.get(), &webgpu_engine::TileMeshRenderer::update_gpu_tiles_ortho);
    QObject::connect(ortho_scheduler.scheduler.get(),    &TextureScheduler::gpu_tiles_updated,  &window,                  &Window::request_render);

    QObject::connect(&window, &Window::mouse_moved,  &camera_controller, &CameraController::mouse_move);
    QObject::connect(&window, &Window::mouse_pressed, &camera_controller, &CameraController::mouse_press);
    QObject::connect(&window, &Window::wheel_turned,  &camera_controller, &CameraController::wheel_turn);
    QObject::connect(&window, &Window::touch_made,    &camera_controller, &CameraController::touch);
    QObject::connect(&window, &Window::key_pressed,   &camera_controller, &CameraController::key_press);
    QObject::connect(&window, &Window::key_released,  &camera_controller, &CameraController::key_release);
    QObject::connect(&window, &Window::resized, &camera_controller, [&camera_controller](glm::uvec2 new_size) { camera_controller.set_viewport(new_size); });
    QObject::connect(&window, &Window::about_to_be_destoryed, context.get(), [&context]() {
        if (context->is_alive())
            context->destroy();
    });
    // clang-format on

    QObject::connect(&app, &QCoreApplication::aboutToQuit, geometry_scheduler.scheduler.get(), [&]() {
        geometry_scheduler.scheduler.reset();
        ortho_scheduler.scheduler.reset();
    });
    QObject::connect(&app, &QCoreApplication::aboutToQuit, geometry_scheduler.tile_service.get(), [&]() {
        geometry_scheduler.tile_service.reset();
        ortho_scheduler.tile_service.reset();
    });
    QObject::connect(&app, &QCoreApplication::aboutToQuit, &app, [&]() {
        scheduler_thread.wait(100);
        scheduler_thread.quit();
        scheduler_thread.wait(500);
    });

    window.show();
    camera_controller.set_viewport({ std::max(1, window.width()), std::max(1, window.height()) });
    scheduler_thread.start();

    return QGuiApplication::exec();
}
