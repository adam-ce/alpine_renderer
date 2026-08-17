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
        m_pending = true;
    }

    void render() override
    {
        if (!m_pending)
            return;
        m_pending = false;
        m_window->beginExternalCommands();
        const auto [text, json] = run();
        m_window->endExternalCommands();
        QPointer<BenchmarkItem> item = m_item;
        QMetaObject::invokeMethod(m_item, [item, text, json]() {
            if (item)
                item->publishResults(text, json);
        });
    }

    QOpenGLFramebufferObject* createFramebufferObject(const QSize&) override
    {
        QOpenGLFramebufferObjectFormat format;
        format.setAttachment(QOpenGLFramebufferObject::NoAttachment);
        return new QOpenGLFramebufferObject(QSize(1, 1), format);
    }

private:
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
                { QStringLiteral("error"), QStringLiteral("Compressed texture arrays require WEBGL_compressed_texture_etc") },
            };
            const auto json = QString::fromUtf8(QJsonDocument(root).toJson(QJsonDocument::Indented));
            qInfo().noquote() << json;
            return { QStringLiteral("GPU compression is unavailable: this WebGL device does not expose WEBGL_compressed_texture_etc."), json };
        }

        const auto algorithm = gl_engine::Texture::compression_algorithm();
        const auto filter = m_mipmaps ? gl_engine::Texture::Filter::MipMapLinear : gl_engine::Texture::Filter::Linear;
        gl_engine::Texture cpu_destination(gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
        cpu_destination.setParams(filter, gl_engine::Texture::Filter::Linear);
        cpu_destination.allocate_array(resolution, resolution, unsigned(m_batch_size));
        gl_engine::Texture gpu_destination(gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
        gpu_destination.setParams(filter, gl_engine::Texture::Filter::Linear);
        gpu_destination.allocate_array(resolution, resolution, unsigned(m_batch_size));
        gl_engine::TextureCompressor gpu_compressor(resolution, resolution, unsigned(m_batch_size));

        auto upload_cpu = [&](const std::vector<nucleus::utils::MipmappedColourTexture>& compressed) {
            for (size_t layer = 0; layer < compressed.size(); ++layer)
                cpu_destination.upload(compressed[layer], unsigned(layer));
            QOpenGLContext::currentContext()->extraFunctions()->glFinish();
        };

        auto warmup_cpu = cpu_compress(sources, algorithm, m_mipmaps);
        upload_cpu(warmup_cpu);
        static_cast<void>(gpu_compressor.compress(sources,
            gpu_destination,
            layers,
            { .algorithm = algorithm, .effort = unsigned(m_effort), .generate_mipmaps = m_mipmaps }));

        std::vector<double> cpu_compression_times;
        std::vector<double> cpu_total_times;
        std::vector<double> gpu_upload_times;
        std::vector<double> gpu_mipmap_times;
        std::vector<double> gpu_encoding_times;
        std::vector<double> gpu_compressed_upload_times;
        std::vector<double> gpu_total_times;
        cpu_compression_times.reserve(size_t(m_iterations));
        cpu_total_times.reserve(size_t(m_iterations));
        gpu_total_times.reserve(size_t(m_iterations));

        for (int iteration = 0; iteration < m_iterations; ++iteration) {
            const auto cpu_start = Clock::now();
            auto compressed = cpu_compress(sources, algorithm, m_mipmaps);
            const auto cpu_compression_time = elapsed_ms(cpu_start);
            upload_cpu(compressed);
            cpu_compression_times.push_back(cpu_compression_time);
            cpu_total_times.push_back(elapsed_ms(cpu_start));

            const auto gpu = gpu_compressor.compress(sources,
                gpu_destination,
                layers,
                { .algorithm = algorithm, .effort = unsigned(m_effort), .generate_mipmaps = m_mipmaps });
            gpu_upload_times.push_back(gpu.timings.scratch_upload_ms);
            gpu_mipmap_times.push_back(gpu.timings.mipmap_generation_ms);
            gpu_encoding_times.push_back(gpu.timings.encoding_ms);
            gpu_compressed_upload_times.push_back(gpu.timings.compressed_upload_ms);
            gpu_total_times.push_back(gpu.timings.total_ms);
        }

        const auto cpu_compression = statistics(cpu_compression_times);
        const auto cpu_total = statistics(cpu_total_times);
        const auto gpu_upload = statistics(gpu_upload_times);
        const auto gpu_mipmap = statistics(gpu_mipmap_times);
        const auto gpu_encoding = statistics(gpu_encoding_times);
        const auto gpu_compressed_upload = statistics(gpu_compressed_upload_times);
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
            { QStringLiteral("timing_method"), QStringLiteral("glFinish-synchronised wall time") },
            { QStringLiteral("resolution"), int(resolution) },
            { QStringLiteral("batch_size"), m_batch_size },
            { QStringLiteral("iterations"), m_iterations },
            { QStringLiteral("effort"), m_effort },
            { QStringLiteral("mipmaps"), m_mipmaps },
            { QStringLiteral("cpu_compression"), to_json(cpu_compression) },
            { QStringLiteral("cpu_end_to_end"), to_json(cpu_total) },
            { QStringLiteral("gpu_scratch_upload"), to_json(gpu_upload) },
            { QStringLiteral("gpu_mipmap_generation"), to_json(gpu_mipmap) },
            { QStringLiteral("gpu_encoding"), to_json(gpu_encoding) },
            { QStringLiteral("gpu_compressed_upload"), to_json(gpu_compressed_upload) },
            { QStringLiteral("gpu_end_to_end"), to_json(gpu_total) },
            { QStringLiteral("cpu_psnr_db"), cpu_psnr },
            { QStringLiteral("gpu_psnr_db"), gpu_psnr },
            { QStringLiteral("cpu_tiles_per_second"), 1000.0 * m_batch_size / cpu_total.median },
            { QStringLiteral("gpu_tiles_per_second"), 1000.0 * m_batch_size / gpu_total.median },
        };
        const auto json = QString::fromUtf8(QJsonDocument(root).toJson(QJsonDocument::Indented));
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
            gl_string(GL_RENDERER),
            QStringLiteral("Timing: completion-synchronised wall time"),
            QString(),
            line(QStringLiteral("CPU compression"), cpu_compression),
            line(QStringLiteral("CPU end-to-end"), cpu_total),
            line(QStringLiteral("GPU scratch upload"), gpu_upload),
            line(QStringLiteral("GPU mip generation"), gpu_mipmap),
            line(QStringLiteral("GPU encoding"), gpu_encoding),
            line(QStringLiteral("GPU compressed upload"), gpu_compressed_upload),
            line(QStringLiteral("GPU end-to-end"), gpu_total),
            QString(),
            QStringLiteral("CPU completed throughput  %1 tiles/s").arg(1000.0 * m_batch_size / cpu_total.median, 0, 'f', 1),
            QStringLiteral("GPU completed throughput  %1 tiles/s").arg(1000.0 * m_batch_size / gpu_total.median, 0, 'f', 1),
            QStringLiteral("CPU PSNR        %1 dB").arg(cpu_psnr, 0, 'f', 2),
            QStringLiteral("GPU PSNR        %1 dB").arg(gpu_psnr, 0, 'f', 2),
        };
        qInfo().noquote() << json;
        return { summary.join('\n'), json };
    }

    QPointer<BenchmarkItem> m_item;
    QQuickWindow* m_window = nullptr;
    unsigned m_seen_serial = 0;
    int m_effort = 4;
    int m_batch_size = 4;
    int m_iterations = 7;
    bool m_mipmaps = true;
    bool m_pending = false;
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
