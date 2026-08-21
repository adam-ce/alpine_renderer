/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#include "TexturePreviewItem.h"

#include <QImage>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFramebufferObjectFormat>
#include <QPainter>
#include <QPointer>
#include <QQuickWindow>
#include <QUrl>
#include <algorithm>
#include <array>
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
#include <nucleus/utils/image_loader.h>

namespace {
using Raster = radix::Raster<glm::u8vec4>;

struct TileGroup {
    int zoom;
    int y;
    int x;
};

constexpr std::array<TileGroup, 16> tile_groups { {
    { 17, 45448, 71496 },
    { 16, 22832, 35144 },
    { 16, 23030, 35578 },
    { 13, 2852, 4476 },
    { 14, 5702, 8808 },
    { 15, 11574, 17670 },
    { 16, 23030, 35078 },
    { 15, 11460, 17622 },
    { 14, 5752, 8656 },
    { 16, 23084, 34746 },
    { 14, 5684, 8926 },
    { 15, 11418, 17692 },
    { 16, 22956, 34570 },
    { 15, 11358, 17904 },
    { 16, 22910, 35770 },
    { 14, 5656, 8938 },
} };

QString tileUrl(const TileGroup& group, int x_offset, int y_offset)
{
    return QStringLiteral("https://gataki.cg.tuwien.ac.at/raw/basemap/tiles/%1/%2/%3.jpeg")
        .arg(group.zoom)
        .arg(group.y + y_offset)
        .arg(group.x + x_offset);
}

double srgbToLinear(uint8_t value)
{
    const auto normalised = double(value) / 255.0;
    if (normalised <= 0.04045)
        return normalised / 12.92;
    return std::pow((normalised + 0.055) / 1.055, 2.4);
}

double linearPsnr(std::span<const QImage> reconstructed, std::span<const Raster> sources)
{
    Q_ASSERT(reconstructed.size() == sources.size());
    double squared_error = 0.0;
    uint64_t channel_count = 0;
    for (size_t i = 0; i < sources.size(); ++i) {
        for (int y = 0; y < reconstructed[i].height(); ++y) {
            for (int x = 0; x < reconstructed[i].width(); ++x) {
                const auto actual = reconstructed[i].pixel(x, y);
                const auto expected = sources[i].pixel({ x, y });
                const std::array<double, 3> actual_channels {
                    qRed(actual) / 255.0,
                    qGreen(actual) / 255.0,
                    qBlue(actual) / 255.0,
                };
                const std::array<double, 3> expected_channels {
                    srgbToLinear(expected.x),
                    srgbToLinear(expected.y),
                    srgbToLinear(expected.z),
                };
                for (size_t channel = 0; channel < actual_channels.size(); ++channel) {
                    const auto difference = actual_channels[channel] - expected_channels[channel];
                    squared_error += difference * difference;
                }
            }
        }
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

constexpr std::array<gl_engine::TextureCompressor::Encoder, 5> gpu_encoders {
    gl_engine::TextureCompressor::Encoder::Search,
    gl_engine::TextureCompressor::Encoder::FastRange,
    gl_engine::TextureCompressor::Encoder::FastSplit,
    gl_engine::TextureCompressor::Encoder::FastSplitFused,
    gl_engine::TextureCompressor::Encoder::FastSplitBounds,
};

QString gpuEncoderName(gl_engine::TextureCompressor::Encoder encoder)
{
    switch (encoder) {
    case gl_engine::TextureCompressor::Encoder::Search:
        return QStringLiteral("GPU Search (reference)");
    case gl_engine::TextureCompressor::Encoder::FastRange:
        return QStringLiteral("GPU Fast range");
    case gl_engine::TextureCompressor::Encoder::FastSplit:
        return QStringLiteral("GPU Fast split");
    case gl_engine::TextureCompressor::Encoder::FastSplitFused:
        return QStringLiteral("GPU Fast split fused");
    case gl_engine::TextureCompressor::Encoder::FastSplitBounds:
        return QStringLiteral("GPU Fast split bounds");
    }
    return {};
}

constexpr size_t uncompressed_preview_index = 0;
constexpr size_t goofy_preview_index = 1;
constexpr size_t first_gpu_preview_index = 2;
constexpr size_t preview_count = first_gpu_preview_index + gpu_encoders.size();
} // namespace

class TexturePreviewRenderer final : public QQuickFramebufferObject::Renderer {
public:
    void synchronize(QQuickFramebufferObject* item) override
    {
        auto* preview_item = static_cast<TexturePreviewItem*>(item);
        m_item = preview_item;
        m_window = preview_item->window();
        m_preview_encoder = preview_item->m_preview_encoder;
        if (preview_item->m_request_serial == m_seen_serial)
            return;
        m_seen_serial = preview_item->m_request_serial;
        m_source_images = preview_item->m_source_images;
        m_pending = true;
    }

    void render() override
    {
        m_window->beginExternalCommands();
        if (m_pending) {
            m_pending = false;
            const auto error = generateTextures();
            QPointer<TexturePreviewItem> item = m_item;
            const auto results = m_preview_results;
            QMetaObject::invokeMethod(m_item, [item, error, results]() {
                if (item)
                    item->publishResults(error, results);
            });
        }
        drawPreview();
        m_window->endExternalCommands();
    }

    QOpenGLFramebufferObject* createFramebufferObject(const QSize& size) override
    {
        QOpenGLFramebufferObjectFormat format;
        format.setAttachment(QOpenGLFramebufferObject::NoAttachment);
        return new QOpenGLFramebufferObject(size.expandedTo(QSize(1, 1)), format);
    }

private:
    QString generateTextures()
    {
        constexpr unsigned resolution = 512;
        constexpr unsigned effort = 4;
        if (m_source_images.size() != tile_groups.size())
            return QStringLiteral("Preview imagery is incomplete.");
        if (!gl_engine::TextureCompressor::is_supported())
            return QStringLiteral("GPU compression is unavailable on this device.");

        std::vector<Raster> sources;
        sources.reserve(m_source_images.size());
        for (const auto& image : m_source_images)
            sources.push_back(nucleus::tile::conversion::to_rgba8raster(image));

        std::vector<unsigned> layers(sources.size());
        std::iota(layers.begin(), layers.end(), 0u);
        const auto algorithm = gl_engine::Texture::compression_algorithm();
        const auto filter = gl_engine::Texture::Filter::MipMapLinear;
        m_preview_results.clear();
        m_preview_results.reserve(preview_count);

        auto create_texture = [&](gl_engine::Texture::Format format) {
            auto texture = std::make_unique<gl_engine::Texture>(gl_engine::Texture::Target::_2dArray, format);
            texture->setParams(format == gl_engine::Texture::Format::CompressedRGBA8 ? filter : gl_engine::Texture::Filter::Linear,
                gl_engine::Texture::Filter::Linear);
            texture->allocate_array(resolution, resolution, unsigned(sources.size()));
            return texture;
        };
        auto psnr = [&](gl_engine::Texture& texture) {
            std::vector<QImage> reconstructed;
            reconstructed.reserve(sources.size());
            for (unsigned layer = 0; layer < sources.size(); ++layer)
                reconstructed.push_back(reconstruct(texture, resolution, layer));
            return linearPsnr(reconstructed, sources);
        };

        m_preview_textures[uncompressed_preview_index] = create_texture(gl_engine::Texture::Format::SRGBA8);
        for (size_t layer = 0; layer < sources.size(); ++layer)
            m_preview_textures[uncompressed_preview_index]->upload(sources[layer], unsigned(layer));
        m_preview_results.push_back({ QStringLiteral("Uncompressed reference"), std::numeric_limits<double>::infinity() });

        m_preview_textures[goofy_preview_index] = create_texture(gl_engine::Texture::Format::CompressedRGBA8);
        for (size_t layer = 0; layer < sources.size(); ++layer) {
            const auto compressed = nucleus::utils::generate_mipmapped_colour_texture(sources[layer], algorithm);
            m_preview_textures[goofy_preview_index]->upload(compressed, unsigned(layer));
        }
        m_preview_results.push_back({ QStringLiteral("Goofy CPU reference"), psnr(*m_preview_textures[goofy_preview_index]) });

        for (size_t i = 0; i < gpu_encoders.size(); ++i) {
            auto& texture = m_preview_textures[first_gpu_preview_index + i];
            texture = create_texture(gl_engine::Texture::Format::CompressedRGBA8);
            gl_engine::TextureCompressor compressor(resolution, resolution, unsigned(sources.size()));
            static_cast<void>(compressor.compress(sources,
                *texture,
                layers,
                {
                    .algorithm = algorithm,
                    .effort = effort,
                    .encoder = gpu_encoders[i],
                    .generate_mipmaps = true,
                }));
            m_preview_results.push_back({ gpuEncoderName(gpu_encoders[i]), psnr(*texture) });
        }
        return {};
    }

    void drawPreview()
    {
        if (m_preview_encoder < 0 || size_t(m_preview_encoder) >= m_preview_textures.size()
            || !m_preview_textures[size_t(m_preview_encoder)])
            return;
        if (!m_preview_shader) {
            m_preview_shader = std::make_unique<gl_engine::ShaderProgram>(R"(
                out highp vec2 texcoords;
                void main() {
                    highp vec2 vertices[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
                    texcoords = 0.5 * gl_Position.xy + vec2(0.5);
                })",
                R"(
                uniform lowp sampler2DArray texture_sampler;
                in highp vec2 texcoords;
                out lowp vec4 out_color;
                highp vec3 linear_to_srgb(highp vec3 linear) {
                    return mix(12.92 * linear,
                        1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055,
                        step(vec3(0.0031308), linear));
                }
                void main() {
                    highp vec2 grid_position = texcoords * 4.0;
                    highp ivec2 cell = min(ivec2(grid_position), ivec2(3));
                    highp float layer = float((3 - cell.y) * 4 + cell.x);
                    highp vec2 tile_coordinates = fract(grid_position);
                    highp vec4 linear_color = textureLod(texture_sampler,
                        vec3(tile_coordinates.x, 1.0 - tile_coordinates.y, layer), 0.0);
                    out_color = vec4(linear_to_srgb(linear_color.rgb), linear_color.a);
                })",
                gl_engine::ShaderCodeSource::PLAINTEXT);
            m_preview_geometry = gl_engine::helpers::create_screen_quad_geometry();
        }

        auto* f = QOpenGLContext::currentContext()->extraFunctions();
        framebufferObject()->bind();
        f->glViewport(0, 0, framebufferObject()->width(), framebufferObject()->height());
        f->glDisable(GL_BLEND);
        f->glDisable(GL_CULL_FACE);
        f->glDisable(GL_DEPTH_TEST);
        f->glDisable(GL_SCISSOR_TEST);
        m_preview_shader->bind();
        m_preview_textures[size_t(m_preview_encoder)]->bind(0);
        m_preview_shader->set_uniform("texture_sampler", 0);
        m_preview_geometry.draw();
        m_preview_shader->release();
    }

    QPointer<TexturePreviewItem> m_item;
    QQuickWindow* m_window = nullptr;
    unsigned m_seen_serial = 0;
    int m_preview_encoder = 0;
    bool m_pending = false;
    std::vector<QImage> m_source_images;
    std::vector<TexturePreviewItem::PreviewResult> m_preview_results;
    std::array<std::unique_ptr<gl_engine::Texture>, preview_count> m_preview_textures;
    std::unique_ptr<gl_engine::ShaderProgram> m_preview_shader;
    gl_engine::helpers::ScreenQuadGeometry m_preview_geometry;
};

TexturePreviewItem::TexturePreviewItem(QQuickItem* parent)
    : QQuickFramebufferObject(parent)
    , m_network_manager(new QNetworkAccessManager(this))
{
    setMirrorVertically(true);
    downloadImages();
}

QQuickFramebufferObject::Renderer* TexturePreviewItem::createRenderer() const { return new TexturePreviewRenderer; }

QString TexturePreviewItem::status() const { return m_status; }
bool TexturePreviewItem::loading() const { return m_loading; }
bool TexturePreviewItem::ready() const { return !m_preview_results.empty(); }
int TexturePreviewItem::previewEncoder() const { return m_preview_encoder; }

void TexturePreviewItem::setPreviewEncoder(int value)
{
    value = m_preview_results.empty() ? 0 : std::clamp(value, 0, int(m_preview_results.size()) - 1);
    if (m_preview_encoder == value)
        return;
    m_preview_encoder = value;
    emit previewEncoderChanged();
    emit previewDetailsChanged();
    update();
}

QStringList TexturePreviewItem::previewEncoders() const
{
    QStringList result;
    result.reserve(qsizetype(m_preview_results.size()));
    for (const auto& preview : m_preview_results)
        result.push_back(preview.name);
    return result;
}

QString TexturePreviewItem::previewName() const
{
    return ready() ? m_preview_results[size_t(m_preview_encoder)].name : QString {};
}

double TexturePreviewItem::previewPsnr() const
{
    return ready() ? m_preview_results[size_t(m_preview_encoder)].psnr : 0.0;
}

void TexturePreviewItem::downloadImages()
{
    m_downloaded_tiles.resize(tile_groups.size() * 4);
    m_downloads_remaining = int(m_downloaded_tiles.size());
    for (size_t group_index = 0; group_index < tile_groups.size(); ++group_index) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                const auto tile_index = group_index * 4 + size_t(y * 2 + x);
                const auto url = tileUrl(tile_groups[group_index], x, y);
                auto* reply = m_network_manager->get(QNetworkRequest(QUrl(url)));
                connect(reply, &QNetworkReply::finished, this, [this, reply, tile_index, url]() {
                    if (reply->error() == QNetworkReply::NoError) {
                        const auto image = nucleus::utils::image_loader::rgba8(reply->readAll());
                        if (image && image->size() == glm::uvec2(256u))
                            m_downloaded_tiles[tile_index] = nucleus::tile::conversion::to_QImage(*image);
                    }
                    if (m_downloaded_tiles[tile_index].isNull())
                        m_status = QStringLiteral("Unable to download preview tile: %1").arg(url);
                    reply->deleteLater();
                    --m_downloads_remaining;
                    if (m_downloads_remaining > 0) {
                        if (!m_status.startsWith(QStringLiteral("Unable"))) {
                            m_status = QStringLiteral("Downloading preview imagery… %1/%2")
                                           .arg(int(m_downloaded_tiles.size()) - m_downloads_remaining)
                                           .arg(m_downloaded_tiles.size());
                        }
                        emit statusChanged();
                        return;
                    }
                    if (std::ranges::any_of(m_downloaded_tiles, [](const QImage& image) { return image.isNull(); })) {
                        m_loading = false;
                        emit statusChanged();
                        return;
                    }
                    stitchImages();
                });
            }
        }
    }
}

void TexturePreviewItem::stitchImages()
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
        m_source_images.push_back(std::move(stitched));
    }
    m_downloaded_tiles.clear();
    m_status = QStringLiteral("Generating compressed texture arrays…");
    ++m_request_serial;
    emit statusChanged();
    update();
}

void TexturePreviewItem::publishResults(const QString& error, const std::vector<PreviewResult>& results)
{
    m_preview_results = results;
    m_preview_encoder = std::clamp(m_preview_encoder, 0, std::max(0, int(m_preview_results.size()) - 1));
    m_status = error.isEmpty() ? QStringLiteral("Texture previews ready.") : error;
    m_loading = false;
    emit statusChanged();
    emit previewResultsChanged();
    emit previewDetailsChanged();
    update();
}
