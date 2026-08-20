import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import TextureCompressionBenchmark

ApplicationWindow {
    id: root

    width: 900
    height: 900
    minimumWidth: 360
    minimumHeight: 640
    visible: true
    title: qsTr("Texture Compression Benchmark")
    LayoutMirroring.enabled: Qt.application.layoutDirection === Qt.RightToLeft
    LayoutMirroring.childrenInherit: true

    BenchmarkItem {
        id: benchmark

        readonly property bool showingPreview: previewDialogLoader.status === Loader.Ready
            && previewDialogLoader.item.opened

        parent: showingPreview ? previewDialogLoader.item.previewHost : root.contentItem
        x: 0
        y: 0
        z: showingPreview ? 0 : -1
        width: showingPreview ? parent.width : 1
        height: showingPreview ? parent.height : 1
    }

    ScrollView {
        id: scrollView
        anchors.fill: parent

        ColumnLayout {
            width: scrollView.availableWidth
            spacing: 16

            Label {
                Layout.fillWidth: true
                Layout.leftMargin: 20
                Layout.rightMargin: 20
                Layout.topMargin: 20
                text: qsTr("CPU versus GPU texture compression")
                font.pointSize: 20
                font.weight: Font.Medium
                wrapMode: Text.Wrap
            }

            Label {
                Layout.fillWidth: true
                Layout.leftMargin: 20
                Layout.rightMargin: 20
                text: qsTr("Compares direct CPU compression, Basis Universal encoding paths, and GPU compression of varied 512×512 ortho imagery in batches of four. Quality is computed once over all 16 images; timing covers three rounds of four distinct batches. WebGL requires ETC or sRGB S3TC compressed textures.")
                wrapMode: Text.Wrap
            }

            GroupBox {
                Layout.fillWidth: true
                Layout.leftMargin: 20
                Layout.rightMargin: 20
                title: qsTr("Configuration")

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 12

                    RowLayout {
                        Layout.fillWidth: true

                        Label {
                            Layout.fillWidth: true
                            text: qsTr("CPU encoder")
                        }

                        ComboBox {
                            Layout.preferredWidth: 260
                            model: [
                                qsTr("Goofy direct (baseline)"),
                                qsTr("BasisU ETC1S"),
                                qsTr("BasisU UASTC LDR 4×4"),
                                qsTr("BasisU XUASTC LDR 4×4")
                            ]
                            currentIndex: benchmark.cpuEncoder
                            enabled: !benchmark.running
                            onActivated: benchmark.cpuEncoder = currentIndex
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        text: qsTr("BasisU quality: %1").arg(benchmark.basisQuality)
                        enabled: benchmark.cpuEncoder !== BenchmarkItem.Goofy
                    }

                    Slider {
                        Layout.fillWidth: true
                        from: 1
                        to: 100
                        stepSize: 1
                        value: benchmark.basisQuality
                        enabled: !benchmark.running && benchmark.cpuEncoder !== BenchmarkItem.Goofy
                        onMoved: benchmark.basisQuality = Math.round(value)
                    }

                    Label {
                        Layout.fillWidth: true
                        text: qsTr("BasisU effort: %1").arg(benchmark.basisEffort)
                        enabled: benchmark.cpuEncoder !== BenchmarkItem.Goofy
                    }

                    Slider {
                        Layout.fillWidth: true
                        from: 0
                        to: 10
                        stepSize: 1
                        value: benchmark.basisEffort
                        enabled: !benchmark.running && benchmark.cpuEncoder !== BenchmarkItem.Goofy
                        onMoved: benchmark.basisEffort = Math.round(value)
                    }

                    Label {
                        Layout.fillWidth: true
                        text: qsTr("GPU effort: %1").arg(benchmark.effort)
                        enabled: benchmark.gpuEncoder === BenchmarkItem.Search
                    }

                    Slider {
                        id: effortSlider
                        Layout.fillWidth: true
                        from: 0
                        to: 10
                        stepSize: 1
                        value: benchmark.effort
                        enabled: !benchmark.running && benchmark.gpuEncoder === BenchmarkItem.Search
                        onMoved: benchmark.effort = Math.round(value)
                    }

                    RowLayout {
                        Layout.fillWidth: true

                        Label {
                            Layout.fillWidth: true
                            text: qsTr("GPU encoder")
                        }

                        ComboBox {
                            Layout.preferredWidth: 260
                            model: [
                                qsTr("Search (reference)"),
                                qsTr("Fast range (Goofy-inspired)"),
                                qsTr("Fast split (two sub-blocks)"),
                                qsTr("Fast split fused"),
                                qsTr("Fast split bounds")
                            ]
                            currentIndex: benchmark.gpuEncoder
                            enabled: !benchmark.running
                            onActivated: benchmark.gpuEncoder = currentIndex
                        }
                    }

                    CheckBox {
                        Layout.fillWidth: true
                        text: qsTr("Generate and compress mipmaps")
                        checked: benchmark.mipmaps
                        enabled: !benchmark.running
                        onToggled: benchmark.mipmaps = checked
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 20
                Layout.rightMargin: 20

                Button {
                    text: qsTr("Run benchmark")
                    enabled: benchmark.dataReady && !benchmark.running
                    highlighted: true
                    onClicked: benchmark.runBenchmark()
                }

                BusyIndicator {
                    Layout.preferredWidth: 44
                    Layout.preferredHeight: 44
                    running: benchmark.running
                    visible: benchmark.running
                }

                Label {
                    Layout.fillWidth: true
                    text: benchmark.running
                        ? qsTr("Computing PSNR, BasisU encoding/transcoding, and batch timings; the display may pause to avoid contaminating GPU measurements.")
                        : benchmark.dataStatus
                    wrapMode: Text.Wrap
                }
            }

            GroupBox {
                Layout.fillWidth: true
                Layout.leftMargin: 20
                Layout.rightMargin: 20
                Layout.bottomMargin: 20
                title: qsTr("Results")

                ColumnLayout {
                    anchors.fill: parent

                    TextArea {
                        Layout.fillWidth: true
                        Layout.minimumHeight: 300
                        text: benchmark.resultText
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.Wrap
                    }

                    RowLayout {
                        Layout.fillWidth: true

                        Button {
                            text: qsTr("Preview encoders")
                            enabled: !benchmark.running && benchmark.previewReady
                            onClicked: {
                                if (previewDialogLoader.status === Loader.Ready)
                                    previewDialogLoader.item.open()
                                else
                                    previewDialogLoader.active = true
                            }
                        }

                        Button {
                            text: qsTr("Copy JSON")
                            enabled: !benchmark.running && benchmark.resultJson.length > 0
                            onClicked: benchmark.copyResultJson()
                        }
                    }
                }
            }
        }
    }

    Loader {
        id: previewDialogLoader

        active: false
        asynchronous: true
        onLoaded: {
            if (status === Loader.Ready)
                item.open()
        }

        sourceComponent: Component {
            Dialog {
                id: previewDialog

                property alias previewHost: previewHost

                parent: Overlay.overlay
                anchors.centerIn: parent
                width: Math.min(root.width - 40, 820)
                height: Math.min(root.height - 40, 880)
                modal: true
                title: qsTr("Compressed tile preview")
                standardButtons: Dialog.Close

                contentItem: ColumnLayout {
                    spacing: 8

                    ComboBox {
                        Layout.fillWidth: true
                        model: benchmark.previewEncoders
                        currentIndex: benchmark.previewEncoder
                        onActivated: benchmark.previewEncoder = currentIndex
                    }

                    Item {
                        id: previewContainer

                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: 240

                        Flickable {
                            id: previewViewport

                            anchors.fill: parent
                            clip: true
                            boundsBehavior: Flickable.StopAtBounds
                            contentWidth: Math.max(width, previewHost.width * previewHost.scale)
                            contentHeight: Math.max(height, previewHost.height * previewHost.scale)

                            Item {
                                id: previewHost

                                readonly property real fittedSize: Math.min(previewViewport.width, previewViewport.height)

                                x: (previewViewport.contentWidth - width) / 2
                                y: (previewViewport.contentHeight - height) / 2
                                width: fittedSize
                                height: width

                                PinchHandler {
                                    target: previewHost
                                    rotationAxis.enabled: false
                                    xAxis.enabled: false
                                    yAxis.enabled: false
                                    scaleAxis.minimum: 1
                                    scaleAxis.maximum: 8
                                }
                            }
                        }

                        Rectangle {
                            anchors.left: parent.left
                            anchors.top: parent.top
                            anchors.margins: 8
                            implicitWidth: previewDetails.implicitWidth + 24
                            implicitHeight: previewDetails.implicitHeight + 24
                            color: Qt.rgba(1, 1, 1, 0.8)
                            radius: 4

                            Label {
                                id: previewDetails

                                readonly property bool showingReference: benchmark.previewCompressionTime < 0

                                anchors.fill: parent
                                anchors.margins: 12
                                text: showingReference
                                    ? qsTr("%1\nPSNR: ∞\nCompression: N/A").arg(benchmark.previewName)
                                    : qsTr("%1\nPSNR: %2 dB\nCompleted compression: %3 ms")
                                        .arg(benchmark.previewName)
                                        .arg(benchmark.previewPsnr.toFixed(2))
                                        .arg(benchmark.previewCompressionTime.toFixed(3))
                            }
                        }
                    }
                }
            }
        }
    }
}
