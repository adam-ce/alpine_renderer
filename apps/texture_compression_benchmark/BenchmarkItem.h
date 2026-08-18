/*****************************************************************************
 * AlpineMaps.org
 * Copyright (C) 2026 Adam Celarek
 * SPDX-License-Identifier: GPL-3.0-or-later
 *****************************************************************************/

#pragma once

#include <QQuickFramebufferObject>
#include <QString>
#include <QtQmlIntegration>

class BenchmarkItem : public QQuickFramebufferObject {
    Q_OBJECT
    QML_ELEMENT
    Q_PROPERTY(int effort READ effort WRITE setEffort NOTIFY effortChanged)
    Q_PROPERTY(int batchSize READ batchSize WRITE setBatchSize NOTIFY batchSizeChanged)
    Q_PROPERTY(int iterations READ iterations WRITE setIterations NOTIFY iterationsChanged)
    Q_PROPERTY(bool mipmaps READ mipmaps WRITE setMipmaps NOTIFY mipmapsChanged)
    Q_PROPERTY(int backend READ backend WRITE setBackend NOTIFY backendChanged)
    Q_PROPERTY(bool transformFeedbackSupported READ transformFeedbackSupported CONSTANT)
    Q_PROPERTY(bool running READ running NOTIFY runningChanged)
    Q_PROPERTY(QString resultText READ resultText NOTIFY resultTextChanged)
    Q_PROPERTY(QString resultJson READ resultJson NOTIFY resultJsonChanged)

public:
    explicit BenchmarkItem(QQuickItem* parent = nullptr);
    Renderer* createRenderer() const override;

    [[nodiscard]] int effort() const;
    void setEffort(int value);
    [[nodiscard]] int batchSize() const;
    void setBatchSize(int value);
    [[nodiscard]] int iterations() const;
    void setIterations(int value);
    [[nodiscard]] bool mipmaps() const;
    void setMipmaps(bool value);
    [[nodiscard]] int backend() const;
    void setBackend(int value);
    [[nodiscard]] bool transformFeedbackSupported() const;
    [[nodiscard]] bool running() const;
    [[nodiscard]] QString resultText() const;
    [[nodiscard]] QString resultJson() const;

    Q_INVOKABLE void runBenchmark();
    Q_INVOKABLE void copyResultJson();

signals:
    void effortChanged();
    void batchSizeChanged();
    void iterationsChanged();
    void mipmapsChanged();
    void backendChanged();
    void runningChanged();
    void resultTextChanged();
    void resultJsonChanged();

private:
    friend class BenchmarkRenderer;
    void publishResults(const QString& text, const QString& json);

    int m_effort = 4;
    int m_batch_size = 4;
    int m_iterations = 10;
    bool m_mipmaps = true;
    int m_backend = 0;
    bool m_running = false;
    unsigned m_request_serial = 0;
    QString m_result_text = QStringLiteral("Run the benchmark to collect results.");
    QString m_result_json;
};
