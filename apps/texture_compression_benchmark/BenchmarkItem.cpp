/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#include "BenchmarkItem.h"

#include <QBuffer>
#include <QClipboard>
#include <QGuiApplication>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFramebufferObjectFormat>
#include <QPointer>
#include <QPainter>
#include <QQuickWindow>
#include <QUrl>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <expected>
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
#include <nucleus/utils/BasisUniversalTextureCompression.h>
#include <nucleus/utils/ColourTexture.h>
#if defined(__EMSCRIPTEN__)
#include <webgl/webgl2.h>
#endif

namespace {
using Raster = radix::Raster<glm::u8vec4>;
using Clock = std::chrono::steady_clock;

struct TileGroup {
    const char* name;
    const char* category;
    int zoom;
    int y;
    int x;
};

constexpr std::array<TileGroup, 16> tile_groups { {
    { "Vienna", "city", 17, 45448, 71496 },
    { "Salzburg", "city", 16, 22832, 35144 },
    { "Graz", "city", 16, 23030, 35578 },
    { "Neusiedler See", "lake", 13, 2852, 4476 },
    { "Attersee", "lake", 14, 5702, 8808 },
    { "Wörthersee", "lake", 15, 11574, 17670 },
    { "Grossglockner", "mountain", 16, 23030, 35078 },
    { "Dachstein", "mountain", 15, 11460, 17622 },
    { "Arlberg", "mountain", 14, 5752, 8656 },
    { "Ötztal", "mountain", 16, 23084, 34746 },
    { "Wienerwald", "forest", 14, 5684, 8926 },
    { "Kalkalpen", "forest", 15, 11418, 17692 },
    { "Bregenzerwald", "forest", 16, 22956, 34570 },
    { "Marchfeld", "fields", 15, 11358, 17904 },
    { "Burgenland", "fields", 16, 22910, 35770 },
    { "Weinviertel", "fields", 14, 5656, 8938 },
} };

QString tile_url(const TileGroup& group, int x_offset, int y_offset)
{
    return QStringLiteral("https://gataki.cg.tuwien.ac.at/raw/basemap/tiles/%1/%2/%3.jpeg")
        .arg(group.zoom)
        .arg(group.y + y_offset)
        .arg(group.x + x_offset);
}

QJsonArray dataset_json()
{
    QJsonArray result;
    for (const auto& group : tile_groups) {
        QJsonArray urls;
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x)
                urls.append(tile_url(group, x, y));
        }
        result.append(QJsonObject {
            { QStringLiteral("name"), QString::fromUtf8(group.name) },
            { QStringLiteral("category"), QString::fromLatin1(group.category) },
            { QStringLiteral("zoom"), group.zoom },
            { QStringLiteral("top_left_x"), group.x },
            { QStringLiteral("top_left_y"), group.y },
            { QStringLiteral("urls"), urls },
        });
    }
    return result;
}

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

QJsonObject size_to_json(const Statistics& value)
{
    return { { QStringLiteral("median_bytes"), value.median },
        { QStringLiteral("p95_bytes"), value.p95 },
        { QStringLiteral("min_bytes"), value.minimum },
        { QStringLiteral("max_bytes"), value.maximum } };
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

struct PendingFenceDiagnostic {
    std::unique_ptr<gl_engine::TextureCompressor> compressor;
    std::unique_ptr<gl_engine::Texture> destination;
    std::unique_ptr<gl_engine::Framebuffer> probe_framebuffer;
    std::unique_ptr<gl_engine::ShaderProgram> probe_shader;
    gl_engine::helpers::ScreenQuadGeometry probe_geometry;
    std::vector<Raster> source_pool;
    std::vector<Raster> sources;
    std::vector<unsigned> layers;
    gl_engine::TextureCompressor::Settings settings;
    GLsync fence = nullptr;
    Clock::time_point started_at;
    int iteration = 0;
    int iteration_count = 0;
    int current_poll_count = 0;
    GLenum wait_error = GL_NO_ERROR;
    double failure_elapsed_ms = 0.0;
    QString status = QStringLiteral("pending");
    std::vector<double> submission_ms;
    std::vector<double> fence_completion_ms;
    std::vector<double> verification_readback_ms;
    std::vector<double> verified_end_to_end_ms;
    std::vector<int> poll_counts;
    std::vector<QString> sample_checksums;
    std::vector<glm::u8vec4> last_source_markers;
    std::vector<glm::u8vec4> last_sampled_pixels;
};

struct PendingGpuReport {
    QJsonObject root;
    QStringList summary;
    std::vector<uint64_t> tickets;
    std::vector<std::optional<gl_engine::TextureCompressor::GpuTimings>> query_results;
    std::vector<bool> query_finished;
    int disjoint_samples = 0;
    std::optional<PendingFenceDiagnostic> fence_diagnostic;
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

double linear_squared_error(const QImage& reconstructed, const Raster& source)
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
    return squared_error;
}

double linear_psnr(std::span<const QImage> reconstructed, std::span<const Raster> sources)
{
    Q_ASSERT(reconstructed.size() == sources.size());
    double squared_error = 0.0;
    uint64_t channel_count = 0;
    for (size_t i = 0; i < sources.size(); ++i) {
        squared_error += linear_squared_error(reconstructed[i], sources[i]);
        channel_count += uint64_t(reconstructed[i].width()) * uint64_t(reconstructed[i].height()) * 3;
    }
    const auto mse = squared_error / double(channel_count);
    return mse == 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(1.0 / mse);
}

QImage reconstruct(gl_engine::Texture& texture, unsigned resolution, unsigned layer)
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
        uniform highp float texture_layer;
        in highp vec2 texcoords;
        out lowp vec4 out_color;
        void main() {
            out_color = textureLod(texture_sampler, vec3(texcoords.x, 1.0 - texcoords.y, texture_layer), 0.0);
        })",
        gl_engine::ShaderCodeSource::PLAINTEXT);
    shader.bind();
    texture.bind(0);
    shader.set_uniform("texture_sampler", 0);
    shader.set_uniform("texture_layer", float(layer));
    gl_engine::helpers::create_screen_quad_geometry().draw();
    auto result = framebuffer.read_colour_attachment(0);
    gl_engine::Framebuffer::unbind();
    return result;
}

QString preview_data_url(std::span<const QImage> images)
{
    constexpr int columns = 4;
    constexpr int tile_size = 512;
    static const auto linear_to_srgb = [] {
        std::array<uint8_t, 256> result {};
        for (size_t i = 0; i < result.size(); ++i) {
            const auto linear = double(i) / 255.0;
            const auto srgb = linear <= 0.0031308 ? 12.92 * linear : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
            result[i] = uint8_t(std::lround(std::clamp(srgb, 0.0, 1.0) * 255.0));
        }
        return result;
    }();
    QImage preview(columns * tile_size, columns * tile_size, QImage::Format_RGBA8888);
    preview.fill(Qt::black);
    QPainter painter(&preview);
    for (size_t i = 0; i < images.size(); ++i) {
        auto srgb_image = images[i].convertToFormat(QImage::Format_RGBA8888);
        for (int y = 0; y < srgb_image.height(); ++y) {
            auto* scanline = srgb_image.scanLine(y);
            for (int x = 0; x < srgb_image.width(); ++x) {
                auto* pixel = scanline + x * 4;
                pixel[0] = linear_to_srgb[pixel[0]];
                pixel[1] = linear_to_srgb[pixel[1]];
                pixel[2] = linear_to_srgb[pixel[2]];
            }
        }
        painter.drawImage(QPoint(int(i % columns) * tile_size, int(i / columns) * tile_size), srgb_image);
    }
    painter.end();

    QByteArray png;
    QBuffer buffer(&png);
    buffer.open(QIODevice::WriteOnly);
    preview.save(&buffer, "PNG");
    return QStringLiteral("data:image/png;base64,") + QString::fromLatin1(png.toBase64());
}

struct CpuCompressionResult {
    std::vector<nucleus::utils::MipmappedColourTexture> textures;
    double source_preparation_ms = 0.0;
    double encoding_ms = 0.0;
    double transcoding_ms = 0.0;
    size_t intermediate_bytes = 0;
    size_t transcoded_bytes = 0;
};

QString cpu_encoder_name(BenchmarkItem::CpuEncoder encoder)
{
    switch (encoder) {
    case BenchmarkItem::CpuEncoder::Goofy:
        return QStringLiteral("Goofy direct");
    case BenchmarkItem::CpuEncoder::BasisEtc1s:
        return QString::fromLatin1(nucleus::utils::basis_universal_format_name(nucleus::utils::BasisUniversalFormat::ETC1S));
    case BenchmarkItem::CpuEncoder::BasisUastcLdr4x4:
        return QString::fromLatin1(nucleus::utils::basis_universal_format_name(nucleus::utils::BasisUniversalFormat::UASTC_LDR_4x4));
    case BenchmarkItem::CpuEncoder::BasisXuastcLdr4x4:
        return QString::fromLatin1(nucleus::utils::basis_universal_format_name(nucleus::utils::BasisUniversalFormat::XUASTC_LDR_4x4));
    }
    return QStringLiteral("Unknown");
}

nucleus::utils::BasisUniversalFormat basis_format(BenchmarkItem::CpuEncoder encoder)
{
    switch (encoder) {
    case BenchmarkItem::CpuEncoder::BasisEtc1s:
        return nucleus::utils::BasisUniversalFormat::ETC1S;
    case BenchmarkItem::CpuEncoder::BasisUastcLdr4x4:
        return nucleus::utils::BasisUniversalFormat::UASTC_LDR_4x4;
    case BenchmarkItem::CpuEncoder::BasisXuastcLdr4x4:
        return nucleus::utils::BasisUniversalFormat::XUASTC_LDR_4x4;
    case BenchmarkItem::CpuEncoder::Goofy:
        break;
    }
    return nucleus::utils::BasisUniversalFormat::ETC1S;
}

std::expected<CpuCompressionResult, QString> cpu_compress(std::span<const Raster> sources,
    nucleus::utils::ColourTexture::Format algorithm,
    bool mipmaps,
    BenchmarkItem::CpuEncoder encoder,
    int basis_quality,
    int basis_effort)
{
    CpuCompressionResult result;
    result.textures.reserve(sources.size());
    for (const auto& source : sources) {
        if (encoder == BenchmarkItem::CpuEncoder::Goofy) {
            const auto start = Clock::now();
            if (mipmaps) {
                result.textures.push_back(nucleus::utils::generate_mipmapped_colour_texture(source, algorithm));
            } else {
                nucleus::utils::MipmappedColourTexture levels;
                levels.emplace_back(source, algorithm);
                result.textures.push_back(std::move(levels));
            }
            result.encoding_ms += elapsed_ms(start);
            for (const auto& level : result.textures.back())
                result.transcoded_bytes += level.n_bytes();
            continue;
        }

        const nucleus::utils::BasisUniversalCompressionSettings settings {
            .format = basis_format(encoder),
            .target_format = algorithm,
            .quality = basis_quality,
            .effort = basis_effort,
            .generate_mipmaps = mipmaps,
        };
        auto compressed = nucleus::utils::compress_with_basis_universal(source, settings);
        if (!compressed)
            return std::unexpected(QString::fromStdString(compressed.error()));
        result.source_preparation_ms += compressed->timings.source_preparation_ms;
        result.encoding_ms += compressed->timings.encoding_ms;
        result.transcoding_ms += compressed->timings.transcoding_ms;
        result.intermediate_bytes += compressed->intermediate_bytes;
        result.transcoded_bytes += compressed->transcoded_bytes;
        result.textures.push_back(std::move(compressed->texture));
    }
    return result;
}

QString gl_string(GLenum name)
{
    const auto* value = QOpenGLContext::currentContext()->functions()->glGetString(name);
    return value ? QString::fromLatin1(reinterpret_cast<const char*>(value)) : QStringLiteral("unavailable");
}

QJsonArray pixels_to_json(const std::vector<glm::u8vec4>& pixels)
{
    QJsonArray result;
    for (const auto& pixel : pixels) {
        result.append(QJsonArray { int(pixel.x), int(pixel.y), int(pixel.z), int(pixel.w) });
    }
    return result;
}

GLsync create_gpu_fence(QOpenGLExtraFunctions* f)
{
#if defined(__EMSCRIPTEN__)
    static_cast<void>(f);
    return emscripten_glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
#else
    return f->glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
#endif
}

void flush_gpu_commands(QOpenGLExtraFunctions* f)
{
    f->glFlush();
}

GLenum poll_gpu_fence(QOpenGLExtraFunctions* f, GLsync fence)
{
#if defined(__EMSCRIPTEN__)
    GLint status = GL_UNSIGNALED;
    f->glGetSynciv(fence, GL_SYNC_STATUS, 1, nullptr, &status);
    return status == GL_SIGNALED ? GL_ALREADY_SIGNALED : GL_TIMEOUT_EXPIRED;
#else
    return f->glClientWaitSync(fence, 0, 0);
#endif
}

void delete_gpu_fence(QOpenGLExtraFunctions* f, GLsync fence)
{
#if defined(__EMSCRIPTEN__)
    static_cast<void>(f);
    emscripten_glDeleteSync(fence);
#else
    f->glDeleteSync(fence);
#endif
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
        m_cpu_encoder = benchmark_item->m_cpu_encoder;
        m_basis_quality = benchmark_item->m_basis_quality;
        m_basis_effort = benchmark_item->m_basis_effort;
        m_gpu_encoder = benchmark_item->m_gpu_encoder;
        m_effort = benchmark_item->m_effort;
        m_mipmaps = benchmark_item->m_mipmaps;
        m_source_images = benchmark_item->m_source_images;
        m_preview_source.clear();
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
            const auto preview_source = m_preview_source;
            QMetaObject::invokeMethod(m_item, [item, text, json, preview_source]() {
                if (item)
                    item->publishResults(text, json, preview_source);
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
        update();
        QPointer<QQuickWindow> window = m_window;
        QMetaObject::invokeMethod(m_window, [window]() {
            if (window)
                window->update();
        });
    }

    void begin_fence_sample(PendingFenceDiagnostic& diagnostic)
    {
        constexpr size_t batch_size = 4;
        const auto group_offset = size_t(diagnostic.iteration % 4) * batch_size;
        diagnostic.sources.assign(
            diagnostic.source_pool.begin() + ptrdiff_t(group_offset), diagnostic.source_pool.begin() + ptrdiff_t(group_offset + batch_size));
        diagnostic.last_source_markers.clear();
        diagnostic.last_source_markers.reserve(diagnostic.sources.size());
        for (size_t layer = 0; layer < diagnostic.sources.size(); ++layer) {
            const auto marker = glm::u8vec4(uint8_t((37 + 53 * layer + 29 * size_t(diagnostic.iteration)) % 256),
                uint8_t((83 + 97 * layer + 47 * size_t(diagnostic.iteration)) % 256),
                uint8_t((149 + 31 * layer + 71 * size_t(diagnostic.iteration)) % 256),
                255);
            diagnostic.last_source_markers.push_back(marker);
            const auto centre = diagnostic.sources[layer].size() / 2u;
            for (unsigned y = centre.y - 8; y < centre.y + 8; ++y) {
                for (unsigned x = centre.x - 8; x < centre.x + 8; ++x)
                    diagnostic.sources[layer].pixel({ x, y }) = marker;
            }
        }

        diagnostic.started_at = Clock::now();
        static_cast<void>(diagnostic.compressor->compress(
            diagnostic.sources, *diagnostic.destination, diagnostic.layers, diagnostic.settings));
        auto* f = QOpenGLContext::currentContext()->extraFunctions();
        diagnostic.fence = create_gpu_fence(f);
        flush_gpu_commands(f);
        diagnostic.submission_ms.push_back(elapsed_ms(diagnostic.started_at));
        diagnostic.current_poll_count = 0;
        if (!diagnostic.fence) {
            diagnostic.status = QStringLiteral("fence_creation_failed");
            diagnostic.wait_error = f->glGetError();
        }
    }

    QImage read_fence_probe(PendingFenceDiagnostic& diagnostic)
    {
        diagnostic.probe_framebuffer->bind();
        diagnostic.probe_shader->bind();
        diagnostic.destination->bind(0);
        diagnostic.probe_shader->set_uniform("texture_sampler", 0);
        diagnostic.probe_geometry.draw();
        auto image = diagnostic.probe_framebuffer->read_colour_attachment(0);
        gl_engine::Framebuffer::unbind();
        return image;
    }

    bool poll_fence_diagnostic(PendingFenceDiagnostic& diagnostic)
    {
        if (diagnostic.status != QStringLiteral("pending"))
            return true;

        ++diagnostic.current_poll_count;
        auto* f = QOpenGLContext::currentContext()->extraFunctions();
        const auto wait_status = poll_gpu_fence(f, diagnostic.fence);
        if (wait_status == GL_TIMEOUT_EXPIRED && elapsed_ms(diagnostic.started_at) < 5000.0)
            return false;
        if (wait_status == GL_TIMEOUT_EXPIRED) {
            diagnostic.status = QStringLiteral("fence_timeout");
            diagnostic.failure_elapsed_ms = elapsed_ms(diagnostic.started_at);
            delete_gpu_fence(f, diagnostic.fence);
            diagnostic.fence = nullptr;
            return true;
        }
        if (wait_status == GL_WAIT_FAILED) {
            diagnostic.status = QStringLiteral("client_wait_failed");
            diagnostic.wait_error = f->glGetError();
            delete_gpu_fence(f, diagnostic.fence);
            diagnostic.fence = nullptr;
            return true;
        }
        if (wait_status != GL_ALREADY_SIGNALED && wait_status != GL_CONDITION_SATISFIED) {
            diagnostic.status = QStringLiteral("unexpected_wait_status");
            diagnostic.wait_error = wait_status;
            delete_gpu_fence(f, diagnostic.fence);
            diagnostic.fence = nullptr;
            return true;
        }

        diagnostic.fence_completion_ms.push_back(elapsed_ms(diagnostic.started_at));
        diagnostic.poll_counts.push_back(diagnostic.current_poll_count);
        delete_gpu_fence(f, diagnostic.fence);
        diagnostic.fence = nullptr;

        const auto readback_start = Clock::now();
        const auto sampled = read_fence_probe(diagnostic);
        diagnostic.verification_readback_ms.push_back(elapsed_ms(readback_start));
        diagnostic.verified_end_to_end_ms.push_back(elapsed_ms(diagnostic.started_at));

        uint64_t checksum = 14695981039346656037ull;
        diagnostic.last_sampled_pixels.clear();
        diagnostic.last_sampled_pixels.reserve(size_t(sampled.width()));
        for (int x = 0; x < sampled.width(); ++x) {
            const auto pixel = sampled.pixel(x, 0);
            const auto rgba = glm::u8vec4(qRed(pixel), qGreen(pixel), qBlue(pixel), qAlpha(pixel));
            diagnostic.last_sampled_pixels.push_back(rgba);
            for (const auto channel : { rgba.x, rgba.y, rgba.z, rgba.w }) {
                checksum ^= channel;
                checksum *= 1099511628211ull;
            }
        }
        diagnostic.sample_checksums.push_back(QStringLiteral("0x%1").arg(checksum, 16, 16, QLatin1Char('0')));

        ++diagnostic.iteration;
        if (diagnostic.iteration < diagnostic.iteration_count) {
            begin_fence_sample(diagnostic);
            return false;
        }
        diagnostic.status = QStringLiteral("valid");
        return true;
    }

    void append_fence_report(PendingGpuReport& report, const PendingFenceDiagnostic& diagnostic)
    {
        QJsonArray poll_counts;
        for (const auto count : diagnostic.poll_counts)
            poll_counts.append(count);
        QJsonArray checksums;
        for (const auto& checksum : diagnostic.sample_checksums)
            checksums.append(checksum);

        QJsonObject json {
            { QStringLiteral("supported"), true },
            { QStringLiteral("status"), diagnostic.status },
            { QStringLiteral("requested_samples"), diagnostic.iteration_count },
            { QStringLiteral("completed_samples"), int(diagnostic.verified_end_to_end_ms.size()) },
            { QStringLiteral("timing_method"),
                QStringLiteral("glFenceSync + later-frame nonblocking status polling; dependent sampled-texture CPU readback") },
            { QStringLiteral("wait_error"), int(diagnostic.wait_error) },
            { QStringLiteral("watchdog_ms"), 5000 },
            { QStringLiteral("failure_elapsed_ms"), diagnostic.failure_elapsed_ms },
            { QStringLiteral("poll_counts"), poll_counts },
            { QStringLiteral("sample_checksums_fnv1a64"), checksums },
            { QStringLiteral("last_source_markers_srgb8"), pixels_to_json(diagnostic.last_source_markers) },
            { QStringLiteral("last_sampled_layers_linear_rgba8"), pixels_to_json(diagnostic.last_sampled_pixels) },
        };
        if (!diagnostic.verified_end_to_end_ms.empty()) {
            json.insert(QStringLiteral("submission"), to_json(statistics(diagnostic.submission_ms)));
            json.insert(QStringLiteral("fence_completion"), to_json(statistics(diagnostic.fence_completion_ms)));
            json.insert(QStringLiteral("verification_readback"), to_json(statistics(diagnostic.verification_readback_ms)));
            json.insert(QStringLiteral("verified_end_to_end"), to_json(statistics(diagnostic.verified_end_to_end_ms)));
            json.insert(QStringLiteral("raw_samples_ms"),
                QJsonObject {
                    { QStringLiteral("submission"), samples_to_json(diagnostic.submission_ms) },
                    { QStringLiteral("fence_completion"), samples_to_json(diagnostic.fence_completion_ms) },
                    { QStringLiteral("verification_readback"), samples_to_json(diagnostic.verification_readback_ms) },
                    { QStringLiteral("verified_end_to_end"), samples_to_json(diagnostic.verified_end_to_end_ms) },
                });
            report.summary.push_back(QString());
            report.summary.push_back(QStringLiteral("Fence + dependent readback verification"));
            report.summary.push_back(QStringLiteral("Fence completion             median %1 ms")
                    .arg(statistics(diagnostic.fence_completion_ms).median, 8, 'f', 3));
            report.summary.push_back(QStringLiteral("Verified end-to-end          median %1 ms")
                    .arg(statistics(diagnostic.verified_end_to_end_ms).median, 8, 'f', 3));
        }
        report.root.insert(QStringLiteral("gpu_fence_verification"), json);
    }

    std::pair<QString, QString> finish_report()
    {
        const auto json = QString::fromUtf8(QJsonDocument(m_pending_gpu_report->root).toJson(QJsonDocument::Indented));
        const auto text = m_pending_gpu_report->summary.join('\n');
        qInfo().noquote() << json;
        m_pending_gpu_report.reset();
        return { text, json };
    }

    std::optional<std::pair<QString, QString>> poll_gpu_report()
    {
        Q_ASSERT(m_pending_gpu_report);
        if (m_pending_gpu_report->fence_diagnostic) {
            if (!poll_fence_diagnostic(*m_pending_gpu_report->fence_diagnostic))
                return std::nullopt;
            append_fence_report(*m_pending_gpu_report, *m_pending_gpu_report->fence_diagnostic);
            m_pending_gpu_report->fence_diagnostic.reset();
        }
        if (!m_gpu_timer->is_supported())
            return finish_report();

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
        return finish_report();
    }

    std::pair<QString, QString> run()
    {
        constexpr unsigned resolution = 512;
        constexpr int batch_size = 4;
        constexpr int batches_per_round = 4;
        constexpr int measurement_rounds = 3;
        constexpr int sample_count = batches_per_round * measurement_rounds;
        if (m_source_images.size() < tile_groups.size()) {
            const QJsonObject error {
                { QStringLiteral("supported"), false },
                { QStringLiteral("error"), QStringLiteral("Benchmark imagery is incomplete") },
            };
            return { QStringLiteral("Unable to load the complete benchmark dataset."),
                QString::fromUtf8(QJsonDocument(error).toJson(QJsonDocument::Indented)) };
        }
        std::vector<Raster> all_sources;
        all_sources.reserve(tile_groups.size());
        for (const auto& image : m_source_images)
            all_sources.push_back(nucleus::tile::conversion::to_rgba8raster(image));
        std::vector<unsigned> layers(size_t(batch_size), 0u);
        std::iota(layers.begin(), layers.end(), 0u);
        std::vector<unsigned> quality_layers(tile_groups.size(), 0u);
        std::iota(quality_layers.begin(), quality_layers.end(), 0u);
        const auto sources_for_group = [&](int group) {
            return std::span<const Raster>(all_sources).subspan(size_t(group * batch_size), size_t(batch_size));
        };

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
        const auto backend_name = [this]() {
            switch (m_gpu_encoder) {
            case BenchmarkItem::GpuEncoder::FastRange:
                return QStringLiteral("Fragment shader fast range + PBO");
            case BenchmarkItem::GpuEncoder::FastSplit:
                return QStringLiteral("Fragment shader fast split + PBO");
            case BenchmarkItem::GpuEncoder::FastSplitFused:
                return QStringLiteral("Fragment shader fast split fused + PBO");
            case BenchmarkItem::GpuEncoder::FastSplitBounds:
                return QStringLiteral("Fragment shader fast split bounds + PBO");
            case BenchmarkItem::GpuEncoder::Search:
                return QStringLiteral("Fragment shader search + PBO");
            }
            return QStringLiteral("Fragment shader search + PBO");
        }();
        const auto filter = m_mipmaps ? gl_engine::Texture::Filter::MipMapLinear : gl_engine::Texture::Filter::Linear;
        const gl_engine::TextureCompressor::Settings gpu_settings {
            .algorithm = algorithm,
            .effort = unsigned(m_effort),
            .encoder = m_gpu_encoder == BenchmarkItem::GpuEncoder::FastRange
                ? gl_engine::TextureCompressor::Encoder::FastRange
                : m_gpu_encoder == BenchmarkItem::GpuEncoder::FastSplit ? gl_engine::TextureCompressor::Encoder::FastSplit
                : m_gpu_encoder == BenchmarkItem::GpuEncoder::FastSplitFused ? gl_engine::TextureCompressor::Encoder::FastSplitFused
                : m_gpu_encoder == BenchmarkItem::GpuEncoder::FastSplitBounds ? gl_engine::TextureCompressor::Encoder::FastSplitBounds
                                                                              : gl_engine::TextureCompressor::Encoder::Search,
            .generate_mipmaps = m_mipmaps,
            .timing_mode = gl_engine::TextureCompressor::TimingMode::EndToEnd,
        };
        const auto compress_cpu = [&](std::span<const Raster> sources) {
            return cpu_compress(sources, algorithm, m_mipmaps, m_cpu_encoder, m_basis_quality, m_basis_effort);
        };
        const auto compression_error = [](const QString& error) {
            const QJsonObject root {
                { QStringLiteral("supported"), false },
                { QStringLiteral("error"), error },
            };
            return std::pair<QString, QString> { QStringLiteral("CPU compression failed: %1").arg(error),
                QString::fromUtf8(QJsonDocument(root).toJson(QJsonDocument::Indented)) };
        };
        auto upload_cpu = [&](gl_engine::Texture& destination, const std::vector<nucleus::utils::MipmappedColourTexture>& compressed) {
            const auto start = Clock::now();
            for (size_t layer = 0; layer < compressed.size(); ++layer)
                destination.upload(compressed[layer], unsigned(layer));
            QOpenGLContext::currentContext()->extraFunctions()->glFinish();
            return elapsed_ms(start);
        };

        // Quality is evaluated first with one untimed compression of all 16 images.
        double cpu_psnr = 0.0;
        double gpu_psnr = 0.0;
        {
            gl_engine::Texture cpu_quality_destination(gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
            cpu_quality_destination.setParams(filter, gl_engine::Texture::Filter::Linear);
            cpu_quality_destination.allocate_array(resolution, resolution, unsigned(tile_groups.size()));
            auto cpu_quality = compress_cpu(all_sources);
            if (!cpu_quality)
                return compression_error(cpu_quality.error());
            static_cast<void>(upload_cpu(cpu_quality_destination, cpu_quality->textures));

            gl_engine::Texture gpu_quality_destination(gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
            gpu_quality_destination.setParams(filter, gl_engine::Texture::Filter::Linear);
            gpu_quality_destination.allocate_array(resolution, resolution, unsigned(tile_groups.size()));
            gl_engine::TextureCompressor gpu_quality_compressor(resolution, resolution, unsigned(tile_groups.size()));
            static_cast<void>(gpu_quality_compressor.compress(all_sources, gpu_quality_destination, quality_layers, gpu_settings));

            std::vector<QImage> cpu_reconstructed;
            std::vector<QImage> gpu_reconstructed;
            cpu_reconstructed.reserve(all_sources.size());
            gpu_reconstructed.reserve(all_sources.size());
            for (unsigned layer = 0; layer < all_sources.size(); ++layer) {
                cpu_reconstructed.push_back(reconstruct(cpu_quality_destination, resolution, layer));
                gpu_reconstructed.push_back(reconstruct(gpu_quality_destination, resolution, layer));
            }
            cpu_psnr = linear_psnr(cpu_reconstructed, all_sources);
            gpu_psnr = linear_psnr(gpu_reconstructed, all_sources);
            m_preview_source = preview_data_url(cpu_reconstructed);
        }

        gl_engine::Texture cpu_destination(gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
        cpu_destination.setParams(filter, gl_engine::Texture::Filter::Linear);
        cpu_destination.allocate_array(resolution, resolution, batch_size);
        auto gpu_destination = std::make_unique<gl_engine::Texture>(
            gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
        gpu_destination->setParams(filter, gl_engine::Texture::Filter::Linear);
        gpu_destination->allocate_array(resolution, resolution, batch_size);
        auto gpu_compressor = std::make_unique<gl_engine::TextureCompressor>(resolution, resolution, batch_size);
        m_gpu_timer = std::make_unique<gl_engine::TextureCompressor::GpuTimer>();

        std::vector<double> cpu_compression_times;
        std::vector<double> cpu_source_preparation_times;
        std::vector<double> cpu_encoding_times;
        std::vector<double> cpu_transcoding_times;
        std::vector<double> cpu_upload_times;
        std::vector<double> cpu_total_times;
        std::vector<double> cpu_intermediate_bytes;
        std::vector<double> cpu_transcoded_bytes;
        WallTimingSamples gpu_upload_times;
        WallTimingSamples gpu_mipmap_times;
        WallTimingSamples gpu_compression_pass_times;
        WallTimingSamples gpu_packing_pass_times;
        WallTimingSamples gpu_encoding_times;
        WallTimingSamples gpu_output_transfer_times;
        WallTimingSamples gpu_compressed_upload_times;
        std::vector<double> gpu_total_times;
        std::vector<uint64_t> gpu_timing_tickets;
        cpu_compression_times.reserve(sample_count);
        cpu_source_preparation_times.reserve(sample_count);
        cpu_encoding_times.reserve(sample_count);
        cpu_transcoding_times.reserve(sample_count);
        cpu_upload_times.reserve(sample_count);
        cpu_total_times.reserve(sample_count);
        cpu_intermediate_bytes.reserve(sample_count);
        cpu_transcoded_bytes.reserve(sample_count);
        gpu_total_times.reserve(sample_count);

        // Keep CPU and GPU phases separate: mobile CPU frequency and thermal state are shared
        // with the GPU, so interleaving them makes the CPU result workload-dependent.
        // Each backend warms up once on all four distinct batches.
        for (int group = 0; group < batches_per_round; ++group) {
            auto compressed = compress_cpu(sources_for_group(group));
            if (!compressed)
                return compression_error(compressed.error());
            static_cast<void>(upload_cpu(cpu_destination, compressed->textures));
        }
        for (int group = 0; group < batches_per_round; ++group)
            static_cast<void>(gpu_compressor->compress(sources_for_group(group), *gpu_destination, layers, gpu_settings));

        for (int round = 0; round < measurement_rounds; ++round) {
            for (int group = 0; group < batches_per_round; ++group) {
                const auto cpu_start = Clock::now();
                auto compressed = compress_cpu(sources_for_group(group));
                if (!compressed)
                    return compression_error(compressed.error());
                const auto cpu_compression_time = elapsed_ms(cpu_start);
                const auto cpu_upload_time = upload_cpu(cpu_destination, compressed->textures);
                cpu_compression_times.push_back(cpu_compression_time);
                cpu_source_preparation_times.push_back(compressed->source_preparation_ms);
                cpu_encoding_times.push_back(compressed->encoding_ms);
                cpu_transcoding_times.push_back(compressed->transcoding_ms);
                cpu_upload_times.push_back(cpu_upload_time);
                cpu_total_times.push_back(elapsed_ms(cpu_start));
                cpu_intermediate_bytes.push_back(double(compressed->intermediate_bytes));
                cpu_transcoded_bytes.push_back(double(compressed->transcoded_bytes));
            }
        }

        for (int round = 0; round < measurement_rounds; ++round) {
            for (int group = 0; group < batches_per_round; ++group) {
                const auto gpu = gpu_compressor->compress(sources_for_group(group), *gpu_destination, layers, gpu_settings);
                gpu_total_times.push_back(gpu.timings.total_ms);
            }
        }

        // Stage timings are collected in a separate profiling phase. Each stage is completed
        // independently, so these values diagnose where time is spent but are not summed to
        // produce the end-to-end result above.
        auto stage_settings = gpu_settings;
        stage_settings.timing_mode = gl_engine::TextureCompressor::TimingMode::IndividualStages;
        stage_settings.gpu_timer = m_gpu_timer.get();
        for (int round = 0; round < measurement_rounds; ++round) {
            for (int group = 0; group < batches_per_round; ++group) {
                const auto gpu = gpu_compressor->compress(sources_for_group(group), *gpu_destination, layers, stage_settings);
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
        }

        const auto cpu_compression = statistics(cpu_compression_times);
        const auto cpu_source_preparation = statistics(cpu_source_preparation_times);
        const auto cpu_encoding = statistics(cpu_encoding_times);
        const auto cpu_transcoding = statistics(cpu_transcoding_times);
        const auto cpu_upload = statistics(cpu_upload_times);
        const auto cpu_total = statistics(cpu_total_times);
        const auto cpu_intermediate_size = statistics(cpu_intermediate_bytes);
        const auto cpu_transcoded_size = statistics(cpu_transcoded_bytes);
        const auto gpu_upload = statistics(gpu_upload_times.total);
        const auto gpu_mipmap = statistics(gpu_mipmap_times.total);
        const auto gpu_compression_pass = statistics(gpu_compression_pass_times.total);
        const auto gpu_packing_pass = statistics(gpu_packing_pass_times.total);
        const auto gpu_encoding = statistics(gpu_encoding_times.total);
        const auto gpu_output_transfer = statistics(gpu_output_transfer_times.total);
        const auto gpu_compressed_upload = statistics(gpu_compressed_upload_times.total);
        const auto gpu_total = statistics(gpu_total_times);
        const auto algorithm_name = algorithm == nucleus::utils::ColourTexture::Format::DXT1 ? QStringLiteral("DXT1 / BC1") : QStringLiteral("ETC1 in ETC2");
        const auto cpu_backend_name = cpu_encoder_name(m_cpu_encoder);

        QJsonObject root {
            { QStringLiteral("renderer"), gl_string(GL_RENDERER) },
            { QStringLiteral("vendor"), gl_string(GL_VENDOR) },
            { QStringLiteral("version"), gl_string(GL_VERSION) },
            { QStringLiteral("supported"), true },
            { QStringLiteral("algorithm"), algorithm_name },
            { QStringLiteral("cpu_backend"), cpu_backend_name },
            { QStringLiteral("gpu_backend"), backend_name },
            { QStringLiteral("timing_method"), QStringLiteral("wall time; one final glFinish per end-to-end sample") },
            { QStringLiteral("resolution"), int(resolution) },
            { QStringLiteral("batch_size"), batch_size },
            { QStringLiteral("measurement_rounds"), measurement_rounds },
            { QStringLiteral("batches_per_round"), batches_per_round },
            { QStringLiteral("measurement_samples"), sample_count },
            { QStringLiteral("warmup_rounds"), 1 },
            { QStringLiteral("gpu_stage_profile_samples"), sample_count },
            { QStringLiteral("basis_quality"), m_basis_quality },
            { QStringLiteral("basis_effort"), m_basis_effort },
            { QStringLiteral("gpu_encoder"), backend_name },
            { QStringLiteral("gpu_effort"), m_effort },
            { QStringLiteral("mipmaps"), m_mipmaps },
            { QStringLiteral("dataset"), dataset_json() },
            { QStringLiteral("cpu_psnr_db_all_16_images"), cpu_psnr },
            { QStringLiteral("gpu_psnr_db_all_16_images"), gpu_psnr },
            { QStringLiteral("cpu_compression"), to_json(cpu_compression) },
            { QStringLiteral("cpu_source_preparation"), to_json(cpu_source_preparation) },
            { QStringLiteral("cpu_encoding"), to_json(cpu_encoding) },
            { QStringLiteral("cpu_transcoding"), to_json(cpu_transcoding) },
            { QStringLiteral("cpu_intermediate_size"), size_to_json(cpu_intermediate_size) },
            { QStringLiteral("cpu_transcoded_size"), size_to_json(cpu_transcoded_size) },
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
            { QStringLiteral("cpu_tiles_per_second"), 1000.0 * batch_size / cpu_total.median },
            { QStringLiteral("gpu_tiles_per_second"), 1000.0 * batch_size / gpu_total.median },
            { QStringLiteral("phase_order"),
                QStringLiteral("16-image PSNR and preview, one CPU and GPU warmup round over four batches, three CPU and GPU measurement rounds over four batches, GPU stage profiling, asynchronous fence verification") },
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
                    { QStringLiteral("cpu_source_preparation"), samples_to_json(cpu_source_preparation_times) },
                    { QStringLiteral("cpu_encoding"), samples_to_json(cpu_encoding_times) },
                    { QStringLiteral("cpu_transcoding"), samples_to_json(cpu_transcoding_times) },
                    { QStringLiteral("cpu_intermediate_bytes"), samples_to_json(cpu_intermediate_bytes) },
                    { QStringLiteral("cpu_transcoded_bytes"), samples_to_json(cpu_transcoded_bytes) },
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
        const auto size_line = [](QString label, Statistics value) {
            return QStringLiteral("%1  median %2 KiB  p95 %3 KiB")
                .arg(label, -26)
                .arg(value.median / 1024.0, 8, 'f', 1)
                .arg(value.p95 / 1024.0, 8, 'f', 1);
        };
        const auto cpu_settings = m_cpu_encoder == BenchmarkItem::CpuEncoder::Goofy
            ? QStringLiteral("CPU: %1").arg(cpu_backend_name)
            : QStringLiteral("CPU: %1, quality %2, effort %3").arg(cpu_backend_name).arg(m_basis_quality).arg(m_basis_effort);
        QStringList summary {
            QStringLiteral("%1 — %2 × %3, batch %4, 12 samples, GPU effort %5, mipmaps %6")
                .arg(algorithm_name)
                .arg(resolution)
                .arg(resolution)
                .arg(batch_size)
                .arg(m_effort)
                .arg(m_mipmaps ? QStringLiteral("on") : QStringLiteral("off")),
            cpu_settings,
            backend_name,
            gl_string(GL_RENDERER),
            QStringLiteral("Quality first: one untimed 16-image pass"),
            QStringLiteral("CPU PSNR (all 16 images)  %1 dB").arg(cpu_psnr, 0, 'f', 2),
            QStringLiteral("GPU PSNR (all 16 images)  %1 dB").arg(gpu_psnr, 0, 'f', 2),
            QStringLiteral("Timing: one warmup round, then three rounds × four distinct batches"),
            QStringLiteral("Timing: one final glFinish per end-to-end sample"),
            QString(),
            line(QStringLiteral("CPU compression"), cpu_compression),
            line(QStringLiteral("  source preparation"), cpu_source_preparation),
            line(QStringLiteral("  CPU encoding"), cpu_encoding),
            line(QStringLiteral("  CPU transcoding"), cpu_transcoding),
            size_line(QStringLiteral("  Intermediate stream"), cpu_intermediate_size),
            size_line(QStringLiteral("  GPU blocks"), cpu_transcoded_size),
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
            QStringLiteral("CPU completed throughput  %1 tiles/s").arg(1000.0 * batch_size / cpu_total.median, 0, 'f', 1),
            QStringLiteral("GPU completed throughput  %1 tiles/s").arg(1000.0 * batch_size / gpu_total.median, 0, 'f', 1),
        };
        if (!m_gpu_timer->is_supported()) {
            root.insert(QStringLiteral("gpu_timer_query"),
                QJsonObject {
                    { QStringLiteral("supported"), false },
                    { QStringLiteral("status"), QStringLiteral("unsupported") },
                });
        }

        m_pending_gpu_report = PendingGpuReport {
            .root = std::move(root),
            .summary = std::move(summary),
            .tickets = std::move(gpu_timing_tickets),
            .query_results = std::vector<std::optional<gl_engine::TextureCompressor::GpuTimings>>(sample_count),
            .query_finished = std::vector<bool>(sample_count, false),
        };
        if (m_gpu_timer->is_supported())
            Q_ASSERT(m_pending_gpu_report->tickets.size() == sample_count);

        PendingFenceDiagnostic fence_diagnostic;
        fence_diagnostic.compressor = std::move(gpu_compressor);
        fence_diagnostic.destination = std::move(gpu_destination);
        fence_diagnostic.probe_framebuffer = std::make_unique<gl_engine::Framebuffer>(gl_engine::Framebuffer::DepthFormat::None,
            std::vector<gl_engine::Framebuffer::ColourFormat> { gl_engine::Framebuffer::ColourFormat::RGBA8 },
            glm::uvec2(batch_size, 1u));
        fence_diagnostic.probe_shader = std::make_unique<gl_engine::ShaderProgram>(R"(
            void main() {
                highp vec2 vertices[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            })",
            R"(
            uniform lowp sampler2DArray texture_sampler;
            out lowp vec4 out_color;
            void main() {
                highp int layer = int(gl_FragCoord.x);
                out_color = textureLod(texture_sampler, vec3(0.5, 0.5, float(layer)), 0.0);
            })",
            gl_engine::ShaderCodeSource::PLAINTEXT);
        fence_diagnostic.probe_geometry = gl_engine::helpers::create_screen_quad_geometry();
        fence_diagnostic.source_pool = std::move(all_sources);
        fence_diagnostic.layers = std::move(layers);
        fence_diagnostic.settings = gpu_settings;
        fence_diagnostic.settings.timing_mode = gl_engine::TextureCompressor::TimingMode::SubmissionOnly;
        fence_diagnostic.iteration_count = sample_count;
        m_pending_gpu_report->fence_diagnostic.emplace(std::move(fence_diagnostic));
        begin_fence_sample(*m_pending_gpu_report->fence_diagnostic);
        return {};
    }

    QPointer<BenchmarkItem> m_item;
    QQuickWindow* m_window = nullptr;
    unsigned m_seen_serial = 0;
    BenchmarkItem::CpuEncoder m_cpu_encoder = BenchmarkItem::CpuEncoder::Goofy;
    int m_basis_quality = 75;
    int m_basis_effort = 4;
    BenchmarkItem::GpuEncoder m_gpu_encoder = BenchmarkItem::GpuEncoder::FastSplitFused;
    int m_effort = 4;
    bool m_mipmaps = true;
    bool m_pending = false;
    std::vector<QImage> m_source_images;
    QString m_preview_source;
    std::unique_ptr<gl_engine::TextureCompressor::GpuTimer> m_gpu_timer;
    std::optional<PendingGpuReport> m_pending_gpu_report;
};

BenchmarkItem::BenchmarkItem(QQuickItem* parent)
    : QQuickFramebufferObject(parent)
    , m_network_manager(new QNetworkAccessManager(this))
{
    downloadBenchmarkData();
}

QQuickFramebufferObject::Renderer* BenchmarkItem::createRenderer() const { return new BenchmarkRenderer; }

BenchmarkItem::CpuEncoder BenchmarkItem::cpuEncoder() const { return m_cpu_encoder; }
void BenchmarkItem::setCpuEncoder(CpuEncoder value)
{
    if (m_cpu_encoder == value)
        return;
    m_cpu_encoder = value;
    emit cpuEncoderChanged();
}

int BenchmarkItem::basisQuality() const { return m_basis_quality; }
void BenchmarkItem::setBasisQuality(int value)
{
    value = std::clamp(value, 1, 100);
    if (m_basis_quality == value)
        return;
    m_basis_quality = value;
    emit basisQualityChanged();
}

int BenchmarkItem::basisEffort() const { return m_basis_effort; }
void BenchmarkItem::setBasisEffort(int value)
{
    value = std::clamp(value, 0, 10);
    if (m_basis_effort == value)
        return;
    m_basis_effort = value;
    emit basisEffortChanged();
}

BenchmarkItem::GpuEncoder BenchmarkItem::gpuEncoder() const { return m_gpu_encoder; }
void BenchmarkItem::setGpuEncoder(GpuEncoder value)
{
    if (m_gpu_encoder == value)
        return;
    m_gpu_encoder = value;
    emit gpuEncoderChanged();
}

int BenchmarkItem::effort() const { return m_effort; }
void BenchmarkItem::setEffort(int value)
{
    value = std::clamp(value, 0, 10);
    if (m_effort == value)
        return;
    m_effort = value;
    emit effortChanged();
}

bool BenchmarkItem::mipmaps() const { return m_mipmaps; }
void BenchmarkItem::setMipmaps(bool value)
{
    if (m_mipmaps == value)
        return;
    m_mipmaps = value;
    emit mipmapsChanged();
}

bool BenchmarkItem::dataReady() const { return m_data_ready; }
QString BenchmarkItem::dataStatus() const { return m_data_status; }
bool BenchmarkItem::running() const { return m_running; }
QString BenchmarkItem::resultText() const { return m_result_text; }
QString BenchmarkItem::resultJson() const { return m_result_json; }
QString BenchmarkItem::previewSource() const { return m_preview_source; }

void BenchmarkItem::downloadBenchmarkData()
{
    m_downloaded_tiles.resize(tile_groups.size() * 4);
    m_downloads_remaining = int(m_downloaded_tiles.size());
    for (size_t group_index = 0; group_index < tile_groups.size(); ++group_index) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                const auto tile_index = group_index * 4 + size_t(y * 2 + x);
                const auto url = tile_url(tile_groups[group_index], x, y);
                auto* reply = m_network_manager->get(QNetworkRequest(QUrl(url)));
                connect(reply, &QNetworkReply::finished, this, [this, reply, tile_index, url]() {
                    if (reply->error() == QNetworkReply::NoError) {
                        QImage image;
                        image.loadFromData(reply->readAll(), "JPEG");
                        if (!image.isNull() && image.size() == QSize(256, 256))
                            m_downloaded_tiles[tile_index] = image.convertToFormat(QImage::Format_RGBA8888);
                    }
                    if (m_downloaded_tiles[tile_index].isNull() && !m_data_status.startsWith(QStringLiteral("Unable"))) {
                        m_data_status = QStringLiteral("Unable to download benchmark tile: %1").arg(url);
                        emit dataStatusChanged();
                    }
                    reply->deleteLater();
                    --m_downloads_remaining;
                    if (m_downloads_remaining > 0) {
                        if (!m_data_status.startsWith(QStringLiteral("Unable"))) {
                            m_data_status = QStringLiteral("Downloading benchmark imagery… %1/%2")
                                                .arg(int(m_downloaded_tiles.size()) - m_downloads_remaining)
                                                .arg(m_downloaded_tiles.size());
                            emit dataStatusChanged();
                        }
                        return;
                    }
                    if (std::ranges::any_of(m_downloaded_tiles, [](const QImage& image) { return image.isNull(); })) {
                        m_result_text = m_data_status;
                        emit resultTextChanged();
                        return;
                    }
                    stitchBenchmarkData();
                });
            }
        }
    }
}

void BenchmarkItem::stitchBenchmarkData()
{
    m_source_images.clear();
    m_source_images.reserve(tile_groups.size());
    for (size_t group_index = 0; group_index < tile_groups.size(); ++group_index) {
        QImage stitched(512, 512, QImage::Format_RGBA8888);
        QPainter painter(&stitched);
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x)
                painter.drawImage(QPoint(x * 256, y * 256), m_downloaded_tiles[group_index * 4 + size_t(y * 2 + x)]);
        }
        painter.end();
        m_source_images.push_back(std::move(stitched));
    }
    m_downloaded_tiles.clear();
    m_data_ready = true;
    m_data_status = QStringLiteral("16 benchmark images ready.");
    emit dataReadyChanged();
    emit dataStatusChanged();
    runBenchmark();
}

void BenchmarkItem::runBenchmark()
{
    if (m_running || !m_data_ready)
        return;
    m_running = true;
    m_result_text = QStringLiteral("Computing PSNR, then measuring batch size 4…");
    m_result_json.clear();
    m_preview_source.clear();
    ++m_request_serial;
    emit runningChanged();
    emit resultTextChanged();
    emit resultJsonChanged();
    emit previewSourceChanged();
    update();
}

void BenchmarkItem::copyResultJson()
{
    if (!m_result_json.isEmpty())
        QGuiApplication::clipboard()->setText(m_result_json);
}

void BenchmarkItem::publishResults(const QString& text, const QString& json, const QString& preview_source)
{
    m_result_text = text;
    m_result_json = json;
    m_preview_source = preview_source;
    m_running = false;
    emit resultTextChanged();
    emit resultJsonChanged();
    emit previewSourceChanged();
    emit runningChanged();
}
