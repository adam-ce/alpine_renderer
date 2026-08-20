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
        x: 0
        y: 0
        z: -1
        width: 1
        height: 1
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
                            text: qsTr("Preview selected CPU encoder")
                            enabled: !benchmark.running && benchmark.previewSource.length > 0
                            onClicked: previewDialogLoader.active = true
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

                parent: Overlay.overlay
                anchors.centerIn: parent
                width: Math.min(root.width - 40, 820)
                height: Math.min(root.height - 40, 880)
                modal: true
                title: qsTr("Selected CPU-compressed tiles")
                standardButtons: Dialog.Close
                onClosed: previewDialogLoader.active = false

                contentItem: Flickable {
                    id: previewViewport

                    property real zoom: 1.0
                    property real pinchStartZoom: 1.0
                    property real pinchStartContentX: 0.0
                    property real pinchStartContentY: 0.0

                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    contentWidth: Math.max(width, previewImage.width)
                    contentHeight: Math.max(height, previewImage.height)

                    Image {
                        id: previewImage

                        readonly property real fittedSize: Math.min(previewViewport.width, previewViewport.height)

                        x: (previewViewport.contentWidth - width) / 2
                        y: (previewViewport.contentHeight - height) / 2
                        width: fittedSize * previewViewport.zoom
                        height: width
                        source: benchmark.previewSource
                        sourceSize: Qt.size(2048, 2048)
                        asynchronous: true
                        fillMode: Image.PreserveAspectFit
                    }

                    PinchHandler {
                        target: null

                        onActiveChanged: {
                            if (active) {
                                previewViewport.pinchStartZoom = previewViewport.zoom
                                previewViewport.pinchStartContentX = previewViewport.contentX
                                previewViewport.pinchStartContentY = previewViewport.contentY
                            } else {
                                previewViewport.returnToBounds()
                            }
                        }
                        onActiveScaleChanged: {
                            const newZoom = Math.max(1.0, Math.min(8.0, previewViewport.pinchStartZoom * activeScale))
                            const zoomRatio = newZoom / previewViewport.pinchStartZoom
                            previewViewport.zoom = newZoom
                            previewViewport.contentX = (previewViewport.pinchStartContentX + centroid.position.x) * zoomRatio - centroid.position.x
                            previewViewport.contentY = (previewViewport.pinchStartContentY + centroid.position.y) * zoomRatio - centroid.position.y
                        }
                    }
                }
            }
        }
    }
}
