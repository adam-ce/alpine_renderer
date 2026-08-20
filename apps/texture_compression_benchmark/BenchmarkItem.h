/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#pragma once

#include <QImage>
#include <QQuickFramebufferObject>
#include <QString>
#include <QtQmlIntegration>
#include <vector>

class QNetworkAccessManager;

class BenchmarkItem : public QQuickFramebufferObject {
    Q_OBJECT
    QML_ELEMENT
    Q_PROPERTY(CpuEncoder cpuEncoder READ cpuEncoder WRITE setCpuEncoder NOTIFY cpuEncoderChanged)
    Q_PROPERTY(int basisQuality READ basisQuality WRITE setBasisQuality NOTIFY basisQualityChanged)
    Q_PROPERTY(int basisEffort READ basisEffort WRITE setBasisEffort NOTIFY basisEffortChanged)
    Q_PROPERTY(GpuEncoder gpuEncoder READ gpuEncoder WRITE setGpuEncoder NOTIFY gpuEncoderChanged)
    Q_PROPERTY(int effort READ effort WRITE setEffort NOTIFY effortChanged)
    Q_PROPERTY(bool mipmaps READ mipmaps WRITE setMipmaps NOTIFY mipmapsChanged)
    Q_PROPERTY(bool dataReady READ dataReady NOTIFY dataReadyChanged)
    Q_PROPERTY(QString dataStatus READ dataStatus NOTIFY dataStatusChanged)
    Q_PROPERTY(bool running READ running NOTIFY runningChanged)
    Q_PROPERTY(QString resultText READ resultText NOTIFY resultTextChanged)
    Q_PROPERTY(QString resultJson READ resultJson NOTIFY resultJsonChanged)
    Q_PROPERTY(QString previewSource READ previewSource NOTIFY previewSourceChanged)

public:
    enum class CpuEncoder { Goofy, BasisEtc1s, BasisUastcLdr4x4, BasisXuastcLdr4x4 };
    Q_ENUM(CpuEncoder)
    enum class GpuEncoder { Search, FastRange, FastSplit, FastSplitFused, FastSplitBounds };
    Q_ENUM(GpuEncoder)

    explicit BenchmarkItem(QQuickItem* parent = nullptr);
    Renderer* createRenderer() const override;

    [[nodiscard]] CpuEncoder cpuEncoder() const;
    void setCpuEncoder(CpuEncoder value);
    [[nodiscard]] int basisQuality() const;
    void setBasisQuality(int value);
    [[nodiscard]] int basisEffort() const;
    void setBasisEffort(int value);
    [[nodiscard]] GpuEncoder gpuEncoder() const;
    void setGpuEncoder(GpuEncoder value);
    [[nodiscard]] int effort() const;
    void setEffort(int value);
    [[nodiscard]] bool mipmaps() const;
    void setMipmaps(bool value);
    [[nodiscard]] bool dataReady() const;
    [[nodiscard]] QString dataStatus() const;
    [[nodiscard]] bool running() const;
    [[nodiscard]] QString resultText() const;
    [[nodiscard]] QString resultJson() const;
    [[nodiscard]] QString previewSource() const;

    Q_INVOKABLE void runBenchmark();
    Q_INVOKABLE void copyResultJson();

signals:
    void cpuEncoderChanged();
    void basisQualityChanged();
    void basisEffortChanged();
    void gpuEncoderChanged();
    void effortChanged();
    void mipmapsChanged();
    void dataReadyChanged();
    void dataStatusChanged();
    void runningChanged();
    void resultTextChanged();
    void resultJsonChanged();
    void previewSourceChanged();

private:
    friend class BenchmarkRenderer;
    void downloadBenchmarkData();
    void stitchBenchmarkData();
    void publishResults(const QString& text, const QString& json, const QString& preview_source);

    CpuEncoder m_cpu_encoder = CpuEncoder::Goofy;
    int m_basis_quality = 75;
    int m_basis_effort = 4;
    GpuEncoder m_gpu_encoder = GpuEncoder::FastSplitFused;
    int m_effort = 4;
    bool m_mipmaps = true;
    bool m_data_ready = false;
    bool m_running = false;
    unsigned m_request_serial = 0;
    int m_downloads_remaining = 0;
    QNetworkAccessManager* m_network_manager = nullptr;
    std::vector<QImage> m_downloaded_tiles;
    std::vector<QImage> m_source_images;
    QString m_data_status = QStringLiteral("Downloading benchmark imagery…");
    QString m_result_text = QStringLiteral("Waiting for benchmark imagery.");
    QString m_result_json;
    QString m_preview_source;
};
