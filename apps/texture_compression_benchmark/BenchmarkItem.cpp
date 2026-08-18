/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#include "BenchmarkItem.h"

#include <QClipboard>
#include <QGuiApplication>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFramebufferObjectFormat>
#include <QPointer>
#include <QQuickWindow>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include <gl_engine/Framebuffer.h>
#include <gl_engine/ShaderProgram.h>
#include <gl_engine/Texture.h>
#include <gl_engine/helpers.h>
#include <nucleus/tile/conversion.h>
#include <nucleus/utils/ColourTexture.h>

namespace {
using Raster = radix::Raster<glm::u8vec4>;
using Clock = std::chrono::steady_clock;

struct Statistics {
    double median = 0.0;
    double p95 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

Statistics statistics(std::vector<double> values)
{
    Q_ASSERT(!values.empty());
    std::ranges::sort(values);
    const auto percentile = [&](double fraction) {
        const auto index = std::min(values.size() - 1, size_t(std::ceil(fraction * double(values.size()))) - 1);
        return values[index];
    };
    return { percentile(0.5), percentile(0.95), values.front(), values.back() };
}

QJsonObject to_json(const Statistics& value)
{
    return { { QStringLiteral("median_ms"), value.median },
        { QStringLiteral("p95_ms"), value.p95 },
        { QStringLiteral("min_ms"), value.minimum },
        { QStringLiteral("max_ms"), value.maximum } };
}

QJsonArray samples_to_json(const std::vector<double>& values)
{
    QJsonArray result;
    for (const auto value : values)
        result.append(value);
    return result;
}

struct WallTimingSamples {
    std::vector<double> total;
    std::vector<double> submission;
    std::vector<double> completion_wait;

    void append(const gl_engine::TextureCompressor::StageTiming& timing)
    {
        total.push_back(timing.total_ms());
        submission.push_back(timing.submission_ms);
        completion_wait.push_back(timing.completion_wait_ms);
    }

    [[nodiscard]] QJsonObject statistics_json() const
    {
        return {
            { QStringLiteral("total"), to_json(statistics(total)) },
            { QStringLiteral("submission"), to_json(statistics(submission)) },
            { QStringLiteral("completion_wait"), to_json(statistics(completion_wait)) },
        };
    }

    [[nodiscard]] QJsonObject raw_json() const
    {
        return {
            { QStringLiteral("total"), samples_to_json(total) },
            { QStringLiteral("submission"), samples_to_json(submission) },
            { QStringLiteral("completion_wait"), samples_to_json(completion_wait) },
        };
    }
};

struct PendingGpuReport {
    QJsonObject root;
    QStringList summary;
    std::vector<uint64_t> tickets;
    std::vector<std::optional<gl_engine::TextureCompressor::GpuTimings>> query_results;
    std::vector<bool> query_finished;
    int disjoint_samples = 0;
};

double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double srgb_to_linear(uint8_t value)
{
    const auto normalised = double(value) / 255.0;
    if (normalised <= 0.04045)
        return normalised / 12.92;
    return std::pow((normalised + 0.055) / 1.055, 2.4);
}

double linear_psnr(const QImage& reconstructed, const Raster& source)
{
    double squared_error = 0.0;
    for (int y = 0; y < reconstructed.height(); ++y) {
        for (int x = 0; x < reconstructed.width(); ++x) {
            const auto actual = reconstructed.pixel(x, y);
            const auto expected = source.pixel({ x, y });
            const std::array<double, 3> actual_channels { qRed(actual) / 255.0, qGreen(actual) / 255.0, qBlue(actual) / 255.0 };
            const std::array<double, 3> expected_channels {
                srgb_to_linear(expected.x), srgb_to_linear(expected.y), srgb_to_linear(expected.z)
            };
            for (size_t channel = 0; channel < actual_channels.size(); ++channel) {
                const auto difference = actual_channels[channel] - expected_channels[channel];
                squared_error += difference * difference;
            }
        }
    }
    const auto mse = squared_error / double(reconstructed.width() * reconstructed.height() * 3);
    return mse == 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(1.0 / mse);
}

QImage reconstruct(gl_engine::Texture& texture, unsigned resolution)
{
    gl_engine::Framebuffer framebuffer(
        gl_engine::Framebuffer::DepthFormat::None, { gl_engine::Framebuffer::ColourFormat::RGBA8 }, { resolution, resolution });
    framebuffer.bind();
    gl_engine::ShaderProgram shader(R"(
        out highp vec2 texcoords;
        void main() {
            vec2 vertices[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            texcoords = 0.5 * gl_Position.xy + vec2(0.5);
        })",
        R"(
        uniform lowp sampler2DArray texture_sampler;
        in highp vec2 texcoords;
        out lowp vec4 out_color;
        void main() {
            out_color = textureLod(texture_sampler, vec3(texcoords.x, 1.0 - texcoords.y, 0.0), 0.0);
        })",
        gl_engine::ShaderCodeSource::PLAINTEXT);
    shader.bind();
    texture.bind(0);
    shader.set_uniform("texture_sampler", 0);
    gl_engine::helpers::create_screen_quad_geometry().draw();
    auto result = framebuffer.read_colour_attachment(0);
    gl_engine::Framebuffer::unbind();
    return result;
}

std::vector<nucleus::utils::MipmappedColourTexture> cpu_compress(
    std::span<const Raster> sources, nucleus::utils::ColourTexture::Format algorithm, bool mipmaps)
{
    std::vector<nucleus::utils::MipmappedColourTexture> result;
    result.reserve(sources.size());
    for (const auto& source : sources) {
        if (mipmaps) {
            result.push_back(nucleus::utils::generate_mipmapped_colour_texture(source, algorithm));
        } else {
            nucleus::utils::MipmappedColourTexture levels;
            levels.emplace_back(source, algorithm);
            result.push_back(std::move(levels));
        }
    }
    return result;
}

QString gl_string(GLenum name)
{
    const auto* value = QOpenGLContext::currentContext()->functions()->glGetString(name);
    return value ? QString::fromLatin1(reinterpret_cast<const char*>(value)) : QStringLiteral("unavailable");
}

} // namespace

class BenchmarkRenderer final : public QQuickFramebufferObject::Renderer {
public:
    void synchronize(QQuickFramebufferObject* item) override
    {
        auto* benchmark_item = static_cast<BenchmarkItem*>(item);
        m_item = benchmark_item;
        m_window = benchmark_item->window();
        if (benchmark_item->m_request_serial == m_seen_serial)
            return;
        m_seen_serial = benchmark_item->m_request_serial;
        m_effort = benchmark_item->m_effort;
        m_batch_size = benchmark_item->m_batch_size;
        m_iterations = benchmark_item->m_iterations;
        m_mipmaps = benchmark_item->m_mipmaps;
        m_backend = benchmark_item->m_backend;
        m_pending = true;
    }

    void render() override
    {
        if (!m_pending && !m_pending_gpu_report)
            return;
        m_window->beginExternalCommands();
        std::optional<std::pair<QString, QString>> completed;
        if (m_pending) {
            m_pending = false;
            const auto immediate = run();
            if (!m_pending_gpu_report)
                completed = immediate;
        } else {
            completed = poll_gpu_report();
        }
        m_window->endExternalCommands();
        if (completed) {
            QPointer<BenchmarkItem> item = m_item;
            const auto [text, json] = std::move(*completed);
            QMetaObject::invokeMethod(m_item, [item, text, json]() {
                if (item)
                    item->publishResults(text, json);
            });
        } else {
            request_another_frame();
        }
    }

    QOpenGLFramebufferObject* createFramebufferObject(const QSize&) override
    {
        QOpenGLFramebufferObjectFormat format;
        format.setAttachment(QOpenGLFramebufferObject::NoAttachment);
        return new QOpenGLFramebufferObject(QSize(1, 1), format);
    }

private:
    void request_another_frame()
    {
        QPointer<BenchmarkItem> item = m_item;
        QMetaObject::invokeMethod(m_item, [item]() {
            if (item)
                item->update();
        });
    }

    std::optional<std::pair<QString, QString>> poll_gpu_report()
    {
        Q_ASSERT(m_pending_gpu_report);
        bool all_finished = true;
        for (size_t i = 0; i < m_pending_gpu_report->tickets.size(); ++i) {
            if (m_pending_gpu_report->query_finished[i])
                continue;
            gl_engine::TextureCompressor::GpuTimings timings;
            const auto status = m_gpu_timer->poll(m_pending_gpu_report->tickets[i], timings);
            if (status == gl_engine::TextureCompressor::GpuTimer::PollStatus::Pending) {
                all_finished = false;
                continue;
            }
            m_pending_gpu_report->query_finished[i] = true;
            if (status == gl_engine::TextureCompressor::GpuTimer::PollStatus::Ready)
                m_pending_gpu_report->query_results[i] = timings;
            else
                ++m_pending_gpu_report->disjoint_samples;
        }
        if (!all_finished)
            return std::nullopt;

        std::vector<double> scratch_upload;
        std::vector<double> mipmap_generation;
        std::vector<double> compression_pass;
        std::vector<double> packing_pass;
        std::vector<double> output_transfer;
        std::vector<double> compressed_upload;
        std::vector<double> total;
        for (const auto& result : m_pending_gpu_report->query_results) {
            if (!result)
                continue;
            scratch_upload.push_back(result->scratch_upload_ms);
            mipmap_generation.push_back(result->mipmap_generation_ms);
            compression_pass.push_back(result->compression_pass_ms);
            packing_pass.push_back(result->packing_pass_ms);
            output_transfer.push_back(result->output_transfer_ms);
            compressed_upload.push_back(result->compressed_upload_ms);
            total.push_back(result->total_ms());
        }

        QJsonObject gpu_timer_json {
            { QStringLiteral("supported"), true },
            { QStringLiteral("timing_method"), QStringLiteral("EXT_disjoint_timer_query; asynchronous GPU elapsed time") },
            { QStringLiteral("requested_samples"), int(m_pending_gpu_report->tickets.size()) },
            { QStringLiteral("valid_samples"), int(total.size()) },
            { QStringLiteral("disjoint_samples"), m_pending_gpu_report->disjoint_samples },
        };
        if (!total.empty()) {
            gpu_timer_json.insert(QStringLiteral("status"),
                m_pending_gpu_report->disjoint_samples ? QStringLiteral("partial") : QStringLiteral("valid"));
            gpu_timer_json.insert(QStringLiteral("scratch_upload"), to_json(statistics(scratch_upload)));
            gpu_timer_json.insert(QStringLiteral("mipmap_generation"), to_json(statistics(mipmap_generation)));
            gpu_timer_json.insert(QStringLiteral("compression_pass"), to_json(statistics(compression_pass)));
            gpu_timer_json.insert(QStringLiteral("packing_pass"), to_json(statistics(packing_pass)));
            gpu_timer_json.insert(QStringLiteral("output_transfer"), to_json(statistics(output_transfer)));
            gpu_timer_json.insert(QStringLiteral("compressed_upload"), to_json(statistics(compressed_upload)));
            gpu_timer_json.insert(QStringLiteral("total_profiled_stages"), to_json(statistics(total)));
            gpu_timer_json.insert(QStringLiteral("raw_samples_ms"),
                QJsonObject {
                    { QStringLiteral("scratch_upload"), samples_to_json(scratch_upload) },
                    { QStringLiteral("mipmap_generation"), samples_to_json(mipmap_generation) },
                    { QStringLiteral("compression_pass"), samples_to_json(compression_pass) },
                    { QStringLiteral("packing_pass"), samples_to_json(packing_pass) },
                    { QStringLiteral("output_transfer"), samples_to_json(output_transfer) },
                    { QStringLiteral("compressed_upload"), samples_to_json(compressed_upload) },
                    { QStringLiteral("total_profiled_stages"), samples_to_json(total) },
                });
            m_pending_gpu_report->summary.push_back(QString());
            m_pending_gpu_report->summary.push_back(QStringLiteral("Actual GPU time (timer query)"));
            m_pending_gpu_report->summary.push_back(
                QStringLiteral("Compression pass            median %1 ms").arg(statistics(compression_pass).median, 8, 'f', 3));
            m_pending_gpu_report->summary.push_back(
                QStringLiteral("Packing pass                median %1 ms").arg(statistics(packing_pass).median, 8, 'f', 3));
            m_pending_gpu_report->summary.push_back(
                QStringLiteral("Profiled GPU stages total   median %1 ms").arg(statistics(total).median, 8, 'f', 3));
        } else {
            gpu_timer_json.insert(QStringLiteral("status"), QStringLiteral("disjoint"));
        }
        m_pending_gpu_report->root.insert(QStringLiteral("gpu_timer_query"), gpu_timer_json);
        const auto json = QString::fromUtf8(QJsonDocument(m_pending_gpu_report->root).toJson(QJsonDocument::Indented));
        const auto text = m_pending_gpu_report->summary.join('\n');
        qInfo().noquote() << json;
        m_pending_gpu_report.reset();
        return std::pair(text, json);
    }

    std::pair<QString, QString> run()
    {
        constexpr unsigned resolution = 512;
        QImage input(QStringLiteral(":/benchmark/merged.jpg"));
        if (input.isNull())
            return { QStringLiteral("Unable to load the benchmark image."), QStringLiteral("{}") };
        input = input.convertToFormat(QImage::Format_RGBA8888).scaled(
            int(resolution), int(resolution), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        const auto source = nucleus::tile::conversion::to_rgba8raster(input);
        std::vector<Raster> sources(size_t(m_batch_size), source);
        std::vector<unsigned> layers(size_t(m_batch_size), 0u);
        std::iota(layers.begin(), layers.end(), 0u);

        if (!gl_engine::TextureCompressor::is_supported()) {
            QJsonObject root {
                { QStringLiteral("renderer"), gl_string(GL_RENDERER) },
                { QStringLiteral("vendor"), gl_string(GL_VENDOR) },
                { QStringLiteral("version"), gl_string(GL_VERSION) },
                { QStringLiteral("supported"), false },
                { QStringLiteral("error"), QStringLiteral("No supported WebGL compressed texture format") },
            };
            const auto json = QString::fromUtf8(QJsonDocument(root).toJson(QJsonDocument::Indented));
            qInfo().noquote() << json;
            return { QStringLiteral("GPU compression is unavailable: this WebGL device exposes neither ETC nor sRGB S3TC."), json };
        }

        const auto algorithm = gl_engine::Texture::compression_algorithm();
        const auto backend = m_backend == 0 ? gl_engine::TextureCompressor::Backend::FragmentShader
                                            : gl_engine::TextureCompressor::Backend::TransformFeedback;
        const auto backend_name = backend == gl_engine::TextureCompressor::Backend::FragmentShader
            ? QStringLiteral("Fragment shader + PBO")
            : QStringLiteral("Transform feedback");
        const auto filter = m_mipmaps ? gl_engine::Texture::Filter::MipMapLinear : gl_engine::Texture::Filter::Linear;
        gl_engine::Texture cpu_destination(gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
        cpu_destination.setParams(filter, gl_engine::Texture::Filter::Linear);
        cpu_destination.allocate_array(resolution, resolution, unsigned(m_batch_size));
        gl_engine::Texture gpu_destination(gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
        gpu_destination.setParams(filter, gl_engine::Texture::Filter::Linear);
        gpu_destination.allocate_array(resolution, resolution, unsigned(m_batch_size));
        gl_engine::TextureCompressor gpu_compressor(resolution, resolution, unsigned(m_batch_size));
        m_gpu_timer = std::make_unique<gl_engine::TextureCompressor::GpuTimer>();

        auto upload_cpu = [&](const std::vector<nucleus::utils::MipmappedColourTexture>& compressed) {
            const auto start = Clock::now();
            for (size_t layer = 0; layer < compressed.size(); ++layer)
                cpu_destination.upload(compressed[layer], unsigned(layer));
            QOpenGLContext::currentContext()->extraFunctions()->glFinish();
            return elapsed_ms(start);
        };

        constexpr int warmup_iterations = 3;
        const gl_engine::TextureCompressor::Settings gpu_settings {
            .algorithm = algorithm,
            .effort = unsigned(m_effort),
            .generate_mipmaps = m_mipmaps,
            .backend = backend,
            .timing_mode = gl_engine::TextureCompressor::TimingMode::EndToEnd,
        };

        std::vector<double> cpu_compression_times;
        std::vector<double> cpu_upload_times;
        std::vector<double> cpu_total_times;
        WallTimingSamples gpu_upload_times;
        WallTimingSamples gpu_mipmap_times;
        WallTimingSamples gpu_compression_pass_times;
        WallTimingSamples gpu_packing_pass_times;
        WallTimingSamples gpu_encoding_times;
        WallTimingSamples gpu_output_transfer_times;
        WallTimingSamples gpu_compressed_upload_times;
        std::vector<double> gpu_total_times;
        std::vector<uint64_t> gpu_timing_tickets;
        cpu_compression_times.reserve(size_t(m_iterations));
        cpu_upload_times.reserve(size_t(m_iterations));
        cpu_total_times.reserve(size_t(m_iterations));
        gpu_total_times.reserve(size_t(m_iterations));

        // Keep CPU and GPU phases separate: mobile CPU frequency and thermal state are shared
        // with the GPU, so interleaving them makes the CPU result backend-dependent.
        for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
            auto compressed = cpu_compress(sources, algorithm, m_mipmaps);
            static_cast<void>(upload_cpu(compressed));
        }
        for (int iteration = 0; iteration < m_iterations; ++iteration) {
            const auto cpu_start = Clock::now();
            auto compressed = cpu_compress(sources, algorithm, m_mipmaps);
            const auto cpu_compression_time = elapsed_ms(cpu_start);
            const auto cpu_upload_time = upload_cpu(compressed);
            cpu_compression_times.push_back(cpu_compression_time);
            cpu_upload_times.push_back(cpu_upload_time);
            cpu_total_times.push_back(elapsed_ms(cpu_start));
        }

        for (int iteration = 0; iteration < warmup_iterations; ++iteration)
            static_cast<void>(gpu_compressor.compress(sources, gpu_destination, layers, gpu_settings));
        for (int iteration = 0; iteration < m_iterations; ++iteration) {
            const auto gpu = gpu_compressor.compress(sources,
                gpu_destination,
                layers,
                gpu_settings);
            gpu_total_times.push_back(gpu.timings.total_ms);
        }

        // Stage timings are collected in a separate profiling phase. Each stage is completed
        // independently, so these values diagnose where time is spent but are not summed to
        // produce the end-to-end result above.
        auto stage_settings = gpu_settings;
        stage_settings.timing_mode = gl_engine::TextureCompressor::TimingMode::IndividualStages;
        stage_settings.gpu_timer = m_gpu_timer.get();
        for (int iteration = 0; iteration < m_iterations; ++iteration) {
            const auto gpu = gpu_compressor.compress(sources, gpu_destination, layers, stage_settings);
            gpu_upload_times.append(gpu.timings.scratch_upload);
            gpu_mipmap_times.append(gpu.timings.mipmap_generation);
            gpu_compression_pass_times.append(gpu.timings.compression_pass);
            gpu_packing_pass_times.append(gpu.timings.packing_pass);
            gpu_encoding_times.append(gpu.timings.encoding);
            gpu_output_transfer_times.append(gpu.timings.output_transfer);
            gpu_compressed_upload_times.append(gpu.timings.compressed_upload);
            if (gpu.gpu_timing_ticket)
                gpu_timing_tickets.push_back(gpu.gpu_timing_ticket);
        }

        const auto cpu_compression = statistics(cpu_compression_times);
        const auto cpu_upload = statistics(cpu_upload_times);
        const auto cpu_total = statistics(cpu_total_times);
        const auto gpu_upload = statistics(gpu_upload_times.total);
        const auto gpu_mipmap = statistics(gpu_mipmap_times.total);
        const auto gpu_compression_pass = statistics(gpu_compression_pass_times.total);
        const auto gpu_packing_pass = statistics(gpu_packing_pass_times.total);
        const auto gpu_encoding = statistics(gpu_encoding_times.total);
        const auto gpu_output_transfer = statistics(gpu_output_transfer_times.total);
        const auto gpu_compressed_upload = statistics(gpu_compressed_upload_times.total);
        const auto gpu_total = statistics(gpu_total_times);
        const auto cpu_psnr = linear_psnr(reconstruct(cpu_destination, resolution), source);
        const auto gpu_psnr = linear_psnr(reconstruct(gpu_destination, resolution), source);
        const auto algorithm_name = algorithm == nucleus::utils::ColourTexture::Format::DXT1 ? QStringLiteral("DXT1 / BC1") : QStringLiteral("ETC1 in ETC2");

        QJsonObject root {
            { QStringLiteral("renderer"), gl_string(GL_RENDERER) },
            { QStringLiteral("vendor"), gl_string(GL_VENDOR) },
            { QStringLiteral("version"), gl_string(GL_VERSION) },
            { QStringLiteral("supported"), true },
            { QStringLiteral("algorithm"), algorithm_name },
            { QStringLiteral("gpu_backend"), backend_name },
            { QStringLiteral("timing_method"), QStringLiteral("wall time; one final glFinish per end-to-end sample") },
            { QStringLiteral("resolution"), int(resolution) },
            { QStringLiteral("batch_size"), m_batch_size },
            { QStringLiteral("iterations"), m_iterations },
            { QStringLiteral("warmup_iterations"), warmup_iterations },
            { QStringLiteral("gpu_stage_profile_iterations"), m_iterations },
            { QStringLiteral("effort"), m_effort },
            { QStringLiteral("mipmaps"), m_mipmaps },
            { QStringLiteral("cpu_compression"), to_json(cpu_compression) },
            { QStringLiteral("cpu_compressed_upload"), to_json(cpu_upload) },
            { QStringLiteral("cpu_end_to_end"), to_json(cpu_total) },
            { QStringLiteral("gpu_scratch_upload"), to_json(gpu_upload) },
            { QStringLiteral("gpu_mipmap_generation"), to_json(gpu_mipmap) },
            { QStringLiteral("gpu_compression_pass"), to_json(gpu_compression_pass) },
            { QStringLiteral("gpu_packing_pass"), to_json(gpu_packing_pass) },
            { QStringLiteral("gpu_encoding"), to_json(gpu_encoding) },
            { QStringLiteral("gpu_output_transfer"), to_json(gpu_output_transfer) },
            { QStringLiteral("gpu_compressed_upload"), to_json(gpu_compressed_upload) },
            { QStringLiteral("gpu_end_to_end"), to_json(gpu_total) },
            { QStringLiteral("cpu_psnr_db"), cpu_psnr },
            { QStringLiteral("gpu_psnr_db"), gpu_psnr },
            { QStringLiteral("cpu_tiles_per_second"), 1000.0 * m_batch_size / cpu_total.median },
            { QStringLiteral("gpu_tiles_per_second"), 1000.0 * m_batch_size / gpu_total.median },
            { QStringLiteral("phase_order"), QStringLiteral("CPU warmup, CPU measurement, GPU warmup, GPU end-to-end measurement, GPU stage profiling") },
            { QStringLiteral("gpu_stage_timing_method"), QStringLiteral("separate profiling pass; each stage glFinish-synchronised") },
            { QStringLiteral("gpu_stage_wall_profile"),
                QJsonObject {
                    { QStringLiteral("scratch_upload"), gpu_upload_times.statistics_json() },
                    { QStringLiteral("mipmap_generation"), gpu_mipmap_times.statistics_json() },
                    { QStringLiteral("compression_pass"), gpu_compression_pass_times.statistics_json() },
                    { QStringLiteral("packing_pass"), gpu_packing_pass_times.statistics_json() },
                    { QStringLiteral("encoding_total"), gpu_encoding_times.statistics_json() },
                    { QStringLiteral("output_transfer"), gpu_output_transfer_times.statistics_json() },
                    { QStringLiteral("compressed_upload"), gpu_compressed_upload_times.statistics_json() },
                    { QStringLiteral("raw_samples_ms"),
                        QJsonObject {
                            { QStringLiteral("scratch_upload"), gpu_upload_times.raw_json() },
                            { QStringLiteral("mipmap_generation"), gpu_mipmap_times.raw_json() },
                            { QStringLiteral("compression_pass"), gpu_compression_pass_times.raw_json() },
                            { QStringLiteral("packing_pass"), gpu_packing_pass_times.raw_json() },
                            { QStringLiteral("encoding_total"), gpu_encoding_times.raw_json() },
                            { QStringLiteral("output_transfer"), gpu_output_transfer_times.raw_json() },
                            { QStringLiteral("compressed_upload"), gpu_compressed_upload_times.raw_json() },
                        } },
                } },
            { QStringLiteral("raw_samples_ms"),
                QJsonObject {
                    { QStringLiteral("cpu_compression"), samples_to_json(cpu_compression_times) },
                    { QStringLiteral("cpu_compressed_upload"), samples_to_json(cpu_upload_times) },
                    { QStringLiteral("cpu_end_to_end"), samples_to_json(cpu_total_times) },
                    { QStringLiteral("gpu_end_to_end"), samples_to_json(gpu_total_times) },
                    { QStringLiteral("gpu_scratch_upload_stage_profile"), samples_to_json(gpu_upload_times.total) },
                    { QStringLiteral("gpu_mipmap_generation_stage_profile"), samples_to_json(gpu_mipmap_times.total) },
                    { QStringLiteral("gpu_compression_pass_stage_profile"), samples_to_json(gpu_compression_pass_times.total) },
                    { QStringLiteral("gpu_packing_pass_stage_profile"), samples_to_json(gpu_packing_pass_times.total) },
                    { QStringLiteral("gpu_encoding_stage_profile"), samples_to_json(gpu_encoding_times.total) },
                    { QStringLiteral("gpu_output_transfer_stage_profile"), samples_to_json(gpu_output_transfer_times.total) },
                    { QStringLiteral("gpu_compressed_upload_stage_profile"), samples_to_json(gpu_compressed_upload_times.total) },
                } },
        };
        const auto line = [](QString label, Statistics value) {
            return QStringLiteral("%1  median %2 ms   p95 %3 ms").arg(label, -26).arg(value.median, 8, 'f', 3).arg(value.p95, 8, 'f', 3);
        };
        QStringList summary {
            QStringLiteral("%1 — %2 × %3, batch %4, effort %5, mipmaps %6")
                .arg(algorithm_name)
                .arg(resolution)
                .arg(resolution)
                .arg(m_batch_size)
                .arg(m_effort)
                .arg(m_mipmaps ? QStringLiteral("on") : QStringLiteral("off")),
            backend_name,
            gl_string(GL_RENDERER),
            QStringLiteral("Timing: one final glFinish per end-to-end sample"),
            QString(),
            line(QStringLiteral("CPU compression"), cpu_compression),
            line(QStringLiteral("CPU compressed upload"), cpu_upload),
            line(QStringLiteral("CPU end-to-end"), cpu_total),
            QStringLiteral("GPU stages (separate serialised profiling pass)"),
            line(QStringLiteral("GPU scratch upload"), gpu_upload),
            line(QStringLiteral("GPU mip generation"), gpu_mipmap),
            line(QStringLiteral("GPU compression pass"), gpu_compression_pass),
            line(QStringLiteral("GPU packing pass"), gpu_packing_pass),
            line(QStringLiteral("GPU encoding"), gpu_encoding),
            line(QStringLiteral("GPU output transfer"), gpu_output_transfer),
            line(QStringLiteral("GPU compressed upload"), gpu_compressed_upload),
            line(QStringLiteral("GPU end-to-end"), gpu_total),
            QString(),
            QStringLiteral("CPU completed throughput  %1 tiles/s").arg(1000.0 * m_batch_size / cpu_total.median, 0, 'f', 1),
            QStringLiteral("GPU completed throughput  %1 tiles/s").arg(1000.0 * m_batch_size / gpu_total.median, 0, 'f', 1),
            QStringLiteral("CPU PSNR        %1 dB").arg(cpu_psnr, 0, 'f', 2),
            QStringLiteral("GPU PSNR        %1 dB").arg(gpu_psnr, 0, 'f', 2),
        };
        if (m_gpu_timer->is_supported()) {
            Q_ASSERT(gpu_timing_tickets.size() == size_t(m_iterations));
            m_pending_gpu_report = PendingGpuReport {
                .root = std::move(root),
                .summary = std::move(summary),
                .tickets = std::move(gpu_timing_tickets),
                .query_results = std::vector<std::optional<gl_engine::TextureCompressor::GpuTimings>>(size_t(m_iterations)),
                .query_finished = std::vector<bool>(size_t(m_iterations), false),
            };
            return {};
        }

        root.insert(QStringLiteral("gpu_timer_query"),
            QJsonObject {
                { QStringLiteral("supported"), false },
                { QStringLiteral("status"), QStringLiteral("unsupported") },
            });
        const auto json = QString::fromUtf8(QJsonDocument(root).toJson(QJsonDocument::Indented));
        const auto text = summary.join('\n');
        qInfo().noquote() << json;
        return { text, json };
    }

    QPointer<BenchmarkItem> m_item;
    QQuickWindow* m_window = nullptr;
    unsigned m_seen_serial = 0;
    int m_effort = 4;
    int m_batch_size = 4;
    int m_iterations = 10;
    bool m_mipmaps = true;
    int m_backend = 0;
    bool m_pending = false;
    std::unique_ptr<gl_engine::TextureCompressor::GpuTimer> m_gpu_timer;
    std::optional<PendingGpuReport> m_pending_gpu_report;
};

BenchmarkItem::BenchmarkItem(QQuickItem* parent)
    : QQuickFramebufferObject(parent)
{
}

QQuickFramebufferObject::Renderer* BenchmarkItem::createRenderer() const { return new BenchmarkRenderer; }

int BenchmarkItem::effort() const { return m_effort; }
void BenchmarkItem::setEffort(int value)
{
    value = std::clamp(value, 0, 10);
    if (m_effort == value)
        return;
    m_effort = value;
    emit effortChanged();
}

int BenchmarkItem::batchSize() const { return m_batch_size; }
void BenchmarkItem::setBatchSize(int value)
{
    if (value != 1 && value != 4 && value != 16)
        return;
    if (m_batch_size == value)
        return;
    m_batch_size = value;
    emit batchSizeChanged();
}

int BenchmarkItem::iterations() const { return m_iterations; }
void BenchmarkItem::setIterations(int value)
{
    value = std::clamp(value, 1, 50);
    if (m_iterations == value)
        return;
    m_iterations = value;
    emit iterationsChanged();
}

bool BenchmarkItem::mipmaps() const { return m_mipmaps; }
void BenchmarkItem::setMipmaps(bool value)
{
    if (m_mipmaps == value)
        return;
    m_mipmaps = value;
    emit mipmapsChanged();
}

int BenchmarkItem::backend() const { return m_backend; }
void BenchmarkItem::setBackend(int value)
{
    if (value < 0 || value > 1 || (value == 1 && !transformFeedbackSupported()))
        return;
    if (m_backend == value)
        return;
    m_backend = value;
    emit backendChanged();
}

bool BenchmarkItem::transformFeedbackSupported() const
{
    return gl_engine::TextureCompressor::is_backend_supported(gl_engine::TextureCompressor::Backend::TransformFeedback);
}

bool BenchmarkItem::running() const { return m_running; }
QString BenchmarkItem::resultText() const { return m_result_text; }
QString BenchmarkItem::resultJson() const { return m_result_json; }

void BenchmarkItem::runBenchmark()
{
    if (m_running)
        return;
    m_running = true;
    ++m_request_serial;
    emit runningChanged();
    update();
}

void BenchmarkItem::copyResultJson()
{
    if (!m_result_json.isEmpty())
        QGuiApplication::clipboard()->setText(m_result_json);
}

void BenchmarkItem::publishResults(const QString& text, const QString& json)
{
    m_result_text = text;
    m_result_json = json;
    m_running = false;
    emit resultTextChanged();
    emit resultJsonChanged();
    emit runningChanged();
}
