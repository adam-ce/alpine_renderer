import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import TextureCompressionBenchmark

ApplicationWindow {
    id: root

    width: 900
    height: 760
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
                text: qsTr("Measures DXT1 or ETC1 compression of 512×512 ortho textures, including optional mipmaps and final texture-array upload. WebGL requires ETC or sRGB S3TC compressed textures.")
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

                    Label {
                        Layout.fillWidth: true
                        text: qsTr("GPU effort: %1").arg(benchmark.effort)
                    }

                    Slider {
                        id: effortSlider
                        Layout.fillWidth: true
                        from: 0
                        to: 10
                        stepSize: 1
                        value: benchmark.effort
                        enabled: !benchmark.running
                        onMoved: benchmark.effort = Math.round(value)
                    }

                    RowLayout {
                        Layout.fillWidth: true

                        Label {
                            Layout.fillWidth: true
                            text: qsTr("Batch size")
                        }

                        ComboBox {
                            id: batchSizeBox
                            Layout.preferredWidth: 120
                            model: [1, 4, 16]
                            currentIndex: 1
                            enabled: !benchmark.running
                            onActivated: benchmark.batchSize = Number(currentText)
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true

                        Label {
                            Layout.fillWidth: true
                            text: qsTr("Measured iterations")
                        }

                        SpinBox {
                            Layout.preferredWidth: 120
                            from: 1
                            to: 50
                            value: benchmark.iterations
                            editable: true
                            enabled: !benchmark.running
                            onValueModified: benchmark.iterations = value
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
                    enabled: !benchmark.running
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
                    text: benchmark.running ? qsTr("Benchmarking; the display may pause to avoid contaminating GPU measurements.") : qsTr("Ready")
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
