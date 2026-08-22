/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#include "TextureCompressionData.h"

#include <QCoreApplication>
#include <QDebug>
#include <QGuiApplication>
#include <QImage>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QOpenGLWindow>
#include <QPainter>
#include <QSurfaceFormat>
#include <QTimer>
#include <QUrl>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <span>
#include <string>
#include <vector>

#include <gl_engine/Framebuffer.h>
#include <gl_engine/ShaderProgram.h>
#include <gl_engine/Texture.h>
#include <gl_engine/helpers.h>
#include <nucleus/tile/conversion.h>
#include <nucleus/utils/image_loader.h>

namespace {
using Clock = std::chrono::steady_clock;
using Raster = radix::Raster<glm::u8vec4>;
using Format = nucleus::utils::ColourTexture::Format;
using Encoder = gl_engine::TextureCompressor::Encoder;

constexpr unsigned resolution = 512;
constexpr unsigned batch_size = 4;
constexpr unsigned framebuffer_size = 32;
constexpr int warmup_batches = 2;
constexpr int measured_batches = 10;
constexpr int repetitions = 200;
constexpr uint32_t random_seed = 0x4a17c0deu;
constexpr int ssim_window_size = 11;
constexpr int ssim_window_radius = ssim_window_size / 2;

enum class Operation { SamplingOnly, Compression };

struct Workload {
    std::array<size_t, batch_size> source_indices {};
    uint32_t sampling_seed = 0;
};

struct QualityMetrics {
    double psnr = 0.0;
    double ssim = 0.0;
};

struct Algorithm {
    std::string name;
    Operation operation = Operation::Compression;
    gl_engine::TextureCompressor::Settings settings;
    bool checksum = false;
    std::vector<double> samples;
    std::optional<QualityMetrics> quality;
};

struct QualityReference {
    std::vector<double> ssim_luma;
    std::vector<double> ssim_mean;
    std::vector<double> ssim_second_moment;
};

struct QualityAccumulator {
    double squared_error = 0.0;
    uint64_t channel_count = 0;
    double ssim_sum = 0.0;
    uint64_t ssim_count = 0;
};

struct Statistics {
    double mean = 0.0;
    double mean_standard_deviation = 0.0;
    size_t sample_count = 0;
};

Statistics mean_statistics(std::span<const double> samples)
{
    if (samples.empty())
        return {};
    Q_ASSERT(samples.size() == size_t(repetitions * measured_batches));
    std::array<double, repetitions> repetition_means {};
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto begin = samples.begin() + repetition * measured_batches;
        repetition_means[size_t(repetition)]
            = std::accumulate(begin, begin + measured_batches, 0.0) / double(measured_batches);
    }

    const auto mean = std::accumulate(repetition_means.begin(), repetition_means.end(), 0.0)
        / double(repetition_means.size());
    double squared_deviations = 0.0;
    for (const auto repetition_mean : repetition_means) {
        const auto deviation = repetition_mean - mean;
        squared_deviations += deviation * deviation;
    }
    const auto repetition_variance = squared_deviations / double(repetition_means.size() - 1);
    const auto mean_standard_deviation = std::sqrt(repetition_variance / double(repetition_means.size()));
    return { mean, mean_standard_deviation, samples.size() };
}

double srgb_to_linear(uint8_t value)
{
    const auto normalised = double(value) / 255.0;
    if (normalised <= 0.04045)
        return normalised / 12.92;
    return std::pow((normalised + 0.055) / 1.055, 2.4);
}

double linear_to_srgb(double value)
{
    if (value <= 0.0031308)
        return 12.92 * value;
    return 1.055 * std::pow(value, 1.0 / 2.4) - 0.055;
}

std::array<double, ssim_window_size> ssim_kernel()
{
    constexpr double sigma = 1.5;
    std::array<double, ssim_window_size> kernel {};
    double sum = 0.0;
    for (int i = -ssim_window_radius; i <= ssim_window_radius; ++i) {
        const auto value = std::exp(-double(i * i) / (2.0 * sigma * sigma));
        kernel[size_t(i + ssim_window_radius)] = value;
        sum += value;
    }
    for (auto& value : kernel)
        value /= sum;
    return kernel;
}

std::vector<double> gaussian_filter_valid(std::span<const double> input, int width, int height)
{
    Q_ASSERT(width >= ssim_window_size && height >= ssim_window_size);
    Q_ASSERT(input.size() == size_t(width * height));
    static const auto kernel = ssim_kernel();
    const auto horizontal_width = width - 2 * ssim_window_radius;
    const auto output_height = height - 2 * ssim_window_radius;
    std::vector<double> horizontal(size_t(horizontal_width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = ssim_window_radius; x < width - ssim_window_radius; ++x) {
            double value = 0.0;
            for (int offset = -ssim_window_radius; offset <= ssim_window_radius; ++offset)
                value += kernel[size_t(offset + ssim_window_radius)] * input[size_t(y * width + x + offset)];
            horizontal[size_t(y * horizontal_width + x - ssim_window_radius)] = value;
        }
    }

    std::vector<double> output(size_t(horizontal_width * output_height));
    for (int y = ssim_window_radius; y < height - ssim_window_radius; ++y) {
        for (int x = 0; x < horizontal_width; ++x) {
            double value = 0.0;
            for (int offset = -ssim_window_radius; offset <= ssim_window_radius; ++offset) {
                value += kernel[size_t(offset + ssim_window_radius)]
                    * horizontal[size_t((y + offset) * horizontal_width + x)];
            }
            output[size_t((y - ssim_window_radius) * horizontal_width + x)] = value;
        }
    }
    return output;
}

QualityReference make_quality_reference(const Raster& source)
{
    QualityReference result;
    result.ssim_luma.resize(size_t(source.width() * source.height()));
    std::vector<double> squared_luma(result.ssim_luma.size());
    for (unsigned y = 0; y < source.height(); ++y) {
        for (unsigned x = 0; x < source.width(); ++x) {
            const auto pixel = source.pixel({ x, y });
            const auto luma = 0.2126 * double(pixel.x) / 255.0
                + 0.7152 * double(pixel.y) / 255.0
                + 0.0722 * double(pixel.z) / 255.0;
            const auto index = size_t(y * source.width() + x);
            result.ssim_luma[index] = luma;
            squared_luma[index] = luma * luma;
        }
    }
    result.ssim_mean = gaussian_filter_valid(result.ssim_luma, int(source.width()), int(source.height()));
    result.ssim_second_moment = gaussian_filter_valid(squared_luma, int(source.width()), int(source.height()));
    return result;
}

void accumulate_quality(QualityAccumulator& accumulator,
    const QImage& reconstructed,
    const Raster& source,
    const QualityReference& reference)
{
    Q_ASSERT(reconstructed.size() == QSize(int(source.width()), int(source.height())));
    std::vector<double> reconstructed_luma(size_t(source.width() * source.height()));
    std::vector<double> squared_luma(reconstructed_luma.size());
    std::vector<double> cross_luma(reconstructed_luma.size());
    for (unsigned y = 0; y < source.height(); ++y) {
        for (unsigned x = 0; x < source.width(); ++x) {
            const auto actual = reconstructed.pixel(int(x), int(y));
            const auto expected = source.pixel({ x, y });
            const std::array<double, 3> actual_linear {
                qRed(actual) / 255.0,
                qGreen(actual) / 255.0,
                qBlue(actual) / 255.0,
            };
            const std::array<double, 3> expected_linear {
                srgb_to_linear(expected.x),
                srgb_to_linear(expected.y),
                srgb_to_linear(expected.z),
            };
            for (size_t channel = 0; channel < actual_linear.size(); ++channel) {
                const auto difference = actual_linear[channel] - expected_linear[channel];
                accumulator.squared_error += difference * difference;
            }
            accumulator.channel_count += actual_linear.size();

            const auto luma = 0.2126 * linear_to_srgb(actual_linear[0])
                + 0.7152 * linear_to_srgb(actual_linear[1])
                + 0.0722 * linear_to_srgb(actual_linear[2]);
            const auto index = size_t(y * source.width() + x);
            reconstructed_luma[index] = luma;
            squared_luma[index] = luma * luma;
            cross_luma[index] = luma * reference.ssim_luma[index];
        }
    }

    const auto actual_mean = gaussian_filter_valid(reconstructed_luma, int(source.width()), int(source.height()));
    const auto actual_second_moment = gaussian_filter_valid(squared_luma, int(source.width()), int(source.height()));
    const auto cross_moment = gaussian_filter_valid(cross_luma, int(source.width()), int(source.height()));
    Q_ASSERT(actual_mean.size() == reference.ssim_mean.size());
    constexpr double c1 = 0.01 * 0.01;
    constexpr double c2 = 0.03 * 0.03;
    for (size_t i = 0; i < actual_mean.size(); ++i) {
        const auto reference_variance
            = std::max(0.0, reference.ssim_second_moment[i] - reference.ssim_mean[i] * reference.ssim_mean[i]);
        const auto actual_variance
            = std::max(0.0, actual_second_moment[i] - actual_mean[i] * actual_mean[i]);
        const auto covariance = cross_moment[i] - reference.ssim_mean[i] * actual_mean[i];
        const auto luminance = 2.0 * reference.ssim_mean[i] * actual_mean[i] + c1;
        const auto contrast_structure = 2.0 * covariance + c2;
        const auto denominator = (reference.ssim_mean[i] * reference.ssim_mean[i] + actual_mean[i] * actual_mean[i] + c1)
            * (reference_variance + actual_variance + c2);
        accumulator.ssim_sum += luminance * contrast_structure / denominator;
    }
    accumulator.ssim_count += actual_mean.size();
}

QualityMetrics quality_metrics(const QualityAccumulator& accumulator)
{
    Q_ASSERT(accumulator.channel_count > 0 && accumulator.ssim_count > 0);
    const auto mse = accumulator.squared_error / double(accumulator.channel_count);
    return {
        mse == 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(1.0 / mse),
        accumulator.ssim_sum / double(accumulator.ssim_count),
    };
}

std::string gl_string(GLenum name)
{
    const auto* value = QOpenGLContext::currentContext()->functions()->glGetString(name);
    return value ? reinterpret_cast<const char*>(value) : "unavailable";
}

const char* format_name(Format format)
{
    switch (format) {
    case Format::DXT1:
        return "DXT1";
    case Format::ETC1:
        return "ETC1";
    case Format::Uncompressed_RGBA:
        return "uncompressed";
    }
    return "unknown";
}

std::vector<Algorithm> supported_algorithms(Format format)
{
    const auto settings = [format](Encoder encoder) {
        return gl_engine::TextureCompressor::Settings {
            .algorithm = format,
            .effort = 0,
            .encoder = encoder,
            .generate_mipmaps = true,
        };
    };

    std::vector<Algorithm> result;
    result.push_back({ "sampling only", Operation::SamplingOnly, settings(Encoder::Checksum) });
    result.push_back({ "checksum", Operation::Compression, settings(Encoder::Checksum), true });
    if (format == Format::DXT1) {
        result.push_back({ "DXT1", Operation::Compression, settings(Encoder::Dxt1) });
    } else if (format == Format::ETC1) {
        result.push_back({ "ETC1 fused exact", Operation::Compression, settings(Encoder::FastSplitFusedExact) });
    }
    return result;
}

class BenchmarkWindow final : public QOpenGLWindow {
public:
    BenchmarkWindow()
        : m_downloaded_tiles(texture_compression_data::tile_groups.size() * 4)
    {
        resize(int(framebuffer_size), int(framebuffer_size));
    }

protected:
    void initializeGL() override
    {
        if (!gl_engine::TextureCompressor::is_supported()) {
            fail(QStringLiteral("GPU texture compression is not supported by this context."));
            return;
        }
        download_data();
    }

    void paintGL() override
    {
        if (!m_data_ready || m_benchmark_started)
            return;
        m_benchmark_started = true;
        const auto successful = run_benchmark();
        QTimer::singleShot(0, qApp, [successful]() { QCoreApplication::exit(successful ? EXIT_SUCCESS : EXIT_FAILURE); });
    }

private:
    void download_data()
    {
        m_downloads_remaining = int(m_downloaded_tiles.size());
        qInfo().noquote() << QStringLiteral("Downloading %1 source tiles once...").arg(m_downloaded_tiles.size());
        for (size_t group_index = 0; group_index < texture_compression_data::tile_groups.size(); ++group_index) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    const auto tile_index = group_index * 4 + size_t(y * 2 + x);
                    const auto url = texture_compression_data::tile_url(
                        texture_compression_data::tile_groups[group_index], x, y);
                    auto* reply = m_network_manager.get(QNetworkRequest(QUrl(url)));
                    connect(reply, &QNetworkReply::finished, this, [this, reply, tile_index, url]() {
                        if (reply->error() == QNetworkReply::NoError) {
                            const auto image = nucleus::utils::image_loader::rgba8(reply->readAll());
                            if (image && image->size() == glm::uvec2(256u))
                                m_downloaded_tiles[tile_index] = nucleus::tile::conversion::to_QImage(*image);
                        }
                        if (m_downloaded_tiles[tile_index].isNull() && m_download_error.isEmpty())
                            m_download_error = QStringLiteral("Unable to download benchmark tile: %1").arg(url);
                        reply->deleteLater();
                        if (--m_downloads_remaining == 0)
                            finish_downloads();
                    });
                }
            }
        }
    }

    void finish_downloads()
    {
        if (!m_download_error.isEmpty()) {
            fail(m_download_error);
            return;
        }

        m_sources.clear();
        m_sources.reserve(texture_compression_data::tile_groups.size());
        for (size_t group_index = 0; group_index < texture_compression_data::tile_groups.size(); ++group_index) {
            QImage stitched(int(resolution), int(resolution), QImage::Format_RGBA8888);
            QPainter painter(&stitched);
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    painter.drawImage(QPoint(x * 256, y * 256),
                        m_downloaded_tiles[group_index * 4 + size_t(y * 2 + x)]);
                }
            }
            painter.end();
            m_sources.push_back(nucleus::tile::conversion::to_rgba8raster(stitched));
        }
        m_downloaded_tiles.clear();
        m_data_ready = true;
        qInfo().noquote() << QStringLiteral("Prepared %1 stitched 512x512 textures.").arg(m_sources.size());
        update();
    }

    bool run_benchmark()
    {
        const auto format = gl_engine::Texture::compression_algorithm();
        auto algorithms = supported_algorithms(format);
        if (algorithms.size() <= 2) {
            qInfo().noquote() << QStringLiteral("No supported GPU compression algorithm was found.");
            return false;
        }

        gl_engine::Texture destination(gl_engine::Texture::Target::_2dArray, gl_engine::Texture::Format::CompressedRGBA8);
        destination.setParams(gl_engine::Texture::Filter::MipMapLinear, gl_engine::Texture::Filter::Linear);
        destination.allocate_array(resolution, resolution, batch_size);
        gl_engine::TextureCompressor compressor(resolution, resolution, batch_size);
        gl_engine::Framebuffer framebuffer(gl_engine::Framebuffer::DepthFormat::None,
            { gl_engine::Framebuffer::ColourFormat::RGBA8 },
            { framebuffer_size, framebuffer_size });
        gl_engine::ShaderProgram sampling_shader(R"(
            void main() {
                highp vec2 vertices[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            })",
            R"(
            uniform lowp sampler2DArray texture_sampler;
            uniform highp uint sampling_seed;
            layout(location = 0) out lowp vec4 out_color;

            highp uint hash(highp uint value) {
                value ^= value >> 16u;
                value *= 0x7feb352du;
                value ^= value >> 15u;
                value *= 0x846ca68bu;
                return value ^ (value >> 16u);
            }

            void main() {
                highp uvec2 pixel = uvec2(gl_FragCoord.xy);
                highp uint pixel_index = pixel.y * 32u + pixel.x;
                highp uint random_value = hash(sampling_seed ^ pixel_index);
                highp float x = float(random_value & 0xffffu) / 65535.0;
                random_value = hash(random_value);
                highp float y = float(random_value & 0xffffu) / 65535.0;
                random_value = hash(random_value);
                highp float layer = float(random_value % 4u);
                random_value = hash(random_value);
                highp float level = float(random_value % 10u);
                lowp vec4 sampled = textureLod(texture_sampler, vec3(x, y, layer), level);
                bool write_pixel = (random_value & 1u) != 0u || pixel_index == 0u;
                if (!write_pixel)
                    discard;
                out_color = sampled;
            })",
            gl_engine::ShaderCodeSource::PLAINTEXT);
        auto sampling_geometry = gl_engine::helpers::create_screen_quad_geometry();
        const std::array<unsigned, batch_size> destination_layers { 0, 1, 2, 3 };

        std::mt19937 random_engine(random_seed);
        std::array<std::array<Workload, warmup_batches + measured_batches>, repetitions> workloads;
        for (auto& repetition : workloads) {
            for (auto& workload : repetition) {
                std::array<size_t, texture_compression_data::tile_groups.size()> indices;
                std::iota(indices.begin(), indices.end(), 0);
                std::ranges::shuffle(indices, random_engine);
                std::ranges::copy_n(indices.begin(), batch_size, workload.source_indices.begin());
                workload.sampling_seed = random_engine();
            }
        }

        const auto run_batch = [&](const Algorithm& algorithm, const Workload& workload) {
            std::vector<Raster> selected_sources;
            if (algorithm.operation == Operation::Compression) {
                selected_sources.reserve(batch_size);
                for (const auto source_index : workload.source_indices)
                    selected_sources.push_back(m_sources[source_index]);
            }

            const auto start = Clock::now();
            if (algorithm.operation == Operation::Compression) {
                static_cast<void>(compressor.compress(
                    selected_sources, destination, destination_layers, algorithm.settings));
            }

            auto* functions = QOpenGLContext::currentContext()->extraFunctions();
            framebuffer.bind();
            functions->glViewport(0, 0, framebuffer_size, framebuffer_size);
            functions->glDisable(GL_BLEND);
            functions->glDisable(GL_CULL_FACE);
            functions->glDisable(GL_DEPTH_TEST);
            functions->glDisable(GL_SCISSOR_TEST);
            functions->glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            sampling_shader.bind();
            destination.bind(0);
            sampling_shader.set_uniform("texture_sampler", 0);
            sampling_shader.set_uniform("sampling_seed", workload.sampling_seed);
            sampling_geometry.draw();
            sampling_shader.release();
            const auto pixel = framebuffer.read_colour_attachment_pixel<glm::u8vec4>(0, { -1.0, -1.0 });
            const auto end = Clock::now();
            m_pixel_checksum += uint64_t(pixel.x) + 3u * uint64_t(pixel.y) + 5u * uint64_t(pixel.z) + 7u * uint64_t(pixel.w);
            return std::chrono::duration<double, std::milli>(end - start).count();
        };

        // Give the sampling-only baseline valid compressed data even if it is first in the random order.
        std::vector<Raster> initial_sources(m_sources.begin(), m_sources.begin() + batch_size);
        static_cast<void>(compressor.compress(
            initial_sources, destination, destination_layers, algorithms[1].settings));

        qInfo().noquote() << QStringLiteral("\nTexture compression benchmark\n"
                                           "GL vendor: %1\n"
                                           "GL renderer: %2\n"
                                           "GL version: %3\n"
                                           "Compression format: %4\n"
                                           "Random seed: 0x%5\n"
                                           "Batch: 4 x 512x512 base-level textures, with mipmaps\n"
                                           "Schedule: %6 repetitions, %7 warm-up + %8 measured batches per algorithm\n"
                                           "Timer: steady-clock wall time through dependent one-pixel framebuffer readback\n")
                                 .arg(QString::fromStdString(gl_string(GL_VENDOR)))
                                 .arg(QString::fromStdString(gl_string(GL_RENDERER)))
                                 .arg(QString::fromStdString(gl_string(GL_VERSION)))
                                 .arg(QString::fromLatin1(format_name(format)))
                                 .arg(QString::number(random_seed, 16))
                                 .arg(repetitions)
                                 .arg(warmup_batches)
                                 .arg(measured_batches);

        std::vector<size_t> algorithm_order(algorithms.size());
        std::iota(algorithm_order.begin(), algorithm_order.end(), 0);
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            std::ranges::shuffle(algorithm_order, random_engine);
            for (const auto algorithm_index : algorithm_order) {
                auto& algorithm = algorithms[algorithm_index];
                for (int batch = 0; batch < warmup_batches; ++batch)
                    static_cast<void>(run_batch(algorithm, workloads[size_t(repetition)][size_t(batch)]));
                for (int batch = 0; batch < measured_batches; ++batch) {
                    algorithm.samples.push_back(run_batch(
                        algorithm, workloads[size_t(repetition)][size_t(warmup_batches + batch)]));
                }
            }
            qInfo().noquote()
                << QStringLiteral("Completed repetition %1/%2").arg(repetition + 1).arg(repetitions);
        }

        qInfo().noquote() << QStringLiteral("Computing untimed 512x512 quality metrics...");
        std::vector<QualityReference> quality_references;
        quality_references.reserve(m_sources.size());
        for (const auto& source : m_sources)
            quality_references.push_back(make_quality_reference(source));

        gl_engine::Framebuffer quality_framebuffer(gl_engine::Framebuffer::DepthFormat::None,
            { gl_engine::Framebuffer::ColourFormat::RGBA8 },
            { resolution, resolution });
        gl_engine::ShaderProgram quality_shader(R"(
            out highp vec2 texcoords;
            void main() {
                highp vec2 vertices[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
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

        const auto reconstruct = [&](unsigned layer) {
            auto* functions = QOpenGLContext::currentContext()->extraFunctions();
            quality_framebuffer.bind();
            functions->glViewport(0, 0, resolution, resolution);
            functions->glDisable(GL_BLEND);
            functions->glDisable(GL_CULL_FACE);
            functions->glDisable(GL_DEPTH_TEST);
            functions->glDisable(GL_SCISSOR_TEST);
            functions->glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            quality_shader.bind();
            destination.bind(0);
            quality_shader.set_uniform("texture_sampler", 0);
            quality_shader.set_uniform("texture_layer", float(layer));
            sampling_geometry.draw();
            quality_shader.release();
            return quality_framebuffer.read_colour_attachment(0);
        };

        for (auto& algorithm : algorithms) {
            if (algorithm.operation != Operation::Compression || algorithm.checksum)
                continue;
            QualityAccumulator accumulator;
            for (size_t source_offset = 0; source_offset < m_sources.size(); source_offset += batch_size) {
                Q_ASSERT(source_offset + batch_size <= m_sources.size());
                std::vector<Raster> selected_sources;
                selected_sources.reserve(batch_size);
                for (size_t layer = 0; layer < batch_size; ++layer)
                    selected_sources.push_back(m_sources[source_offset + layer]);
                static_cast<void>(compressor.compress(
                    selected_sources, destination, destination_layers, algorithm.settings));
                for (unsigned layer = 0; layer < batch_size; ++layer) {
                    accumulate_quality(accumulator,
                        reconstruct(layer),
                        selected_sources[layer],
                        quality_references[source_offset + layer]);
                }
            }
            algorithm.quality = quality_metrics(accumulator);
            qInfo().noquote() << QStringLiteral("Completed quality metrics for %1").arg(QString::fromStdString(algorithm.name));
        }
        gl_engine::Framebuffer::unbind();

        const auto sampling_iterator = std::ranges::find_if(
            algorithms, [](const Algorithm& algorithm) { return algorithm.operation == Operation::SamplingOnly; });
        const auto checksum_iterator = std::ranges::find_if(
            algorithms, [](const Algorithm& algorithm) { return algorithm.checksum; });
        if (sampling_iterator == algorithms.end() || checksum_iterator == algorithms.end())
            return false;

        std::ostringstream report;
        report << "\nAll values are milliseconds per batch. Mean SD is estimated from " << repetitions
               << " repetition means (" << measured_batches << " batches each): sample SD / sqrt("
               << repetitions << ").\n"
               << "Quality is an untimed pass over all " << m_sources.size()
               << " source textures at mip level 0 (512x512). PSNR uses linear RGB; SSIM uses sRGB luma.\n"
               << std::left << std::setw(29) << "algorithm"
               << std::right << std::setw(12) << "raw mean" << std::setw(14) << "raw mean SD"
               << std::setw(8) << "n" << std::setw(17) << "minus sample"
               << std::setw(18) << "adjusted mean SD" << std::setw(17) << "encoding only"
               << std::setw(18) << "encoding mean SD" << std::setw(13) << "PSNR (dB)"
               << std::setw(11) << "SSIM" << '\n';

        for (const auto& algorithm : algorithms) {
            std::vector<double> sampling_subtracted;
            std::vector<double> encoding_only;
            sampling_subtracted.reserve(algorithm.samples.size());
            encoding_only.reserve(algorithm.samples.size());
            for (size_t i = 0; i < algorithm.samples.size(); ++i) {
                sampling_subtracted.push_back(algorithm.samples[i] - sampling_iterator->samples[i]);
                if (algorithm.operation != Operation::SamplingOnly)
                    encoding_only.push_back(algorithm.samples[i] - checksum_iterator->samples[i]);
            }
            const auto raw = mean_statistics(algorithm.samples);
            const auto adjusted = mean_statistics(sampling_subtracted);
            const auto encoding = mean_statistics(encoding_only);
            report << std::left << std::setw(29) << algorithm.name << std::right << std::fixed << std::setprecision(3)
                   << std::setw(12) << raw.mean << std::setw(14) << raw.mean_standard_deviation
                   << std::setw(8) << raw.sample_count << std::setw(17) << adjusted.mean
                   << std::setw(18) << adjusted.mean_standard_deviation;
            if (encoding_only.empty()) {
                report << std::setw(17) << "n/a" << std::setw(18) << "n/a";
            } else {
                report << std::setw(17) << encoding.mean << std::setw(18) << encoding.mean_standard_deviation;
            }
            if (algorithm.quality) {
                report << std::setw(13) << std::setprecision(3) << algorithm.quality->psnr
                       << std::setw(11) << std::setprecision(6) << algorithm.quality->ssim
                       << std::setprecision(3);
            } else {
                report << std::setw(13) << "n/a" << std::setw(11) << "n/a";
            }
            report << '\n';
        }
        report << "Readback checksum: " << m_pixel_checksum;
        for (const auto& line : QString::fromStdString(report.str()).split('\n'))
            qInfo().noquote() << line;
        return true;
    }

    void fail(const QString& message)
    {
        qInfo().noquote() << message;
        QTimer::singleShot(0, qApp, []() { QCoreApplication::exit(EXIT_FAILURE); });
    }

    QNetworkAccessManager m_network_manager;
    std::vector<QImage> m_downloaded_tiles;
    std::vector<Raster> m_sources;
    QString m_download_error;
    int m_downloads_remaining = 0;
    bool m_data_ready = false;
    bool m_benchmark_started = false;
    uint64_t m_pixel_checksum = 0;
};

} // namespace

int main(int argc, char* argv[])
{
    QGuiApplication application(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("TextureCompressionBenchmark"));
    QCoreApplication::setOrganizationName(QStringLiteral("AlpineMaps.org"));

    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setOption(QSurfaceFormat::DebugContext);
    if (QOpenGLContext::openGLModuleType() == QOpenGLContext::LibGL) {
        format.setVersion(3, 3);
        format.setProfile(QSurfaceFormat::CoreProfile);
    } else {
        format.setVersion(3, 0);
    }
    QSurfaceFormat::setDefaultFormat(format);

    BenchmarkWindow window;
    window.show();
    return application.exec();
}
