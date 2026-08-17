#############################################################################
# AlpineMaps.org
# Copyright (C) 2026 Adam Celarek
# SPDX-License-Identifier: GPL-3.0-or-later
#############################################################################

project(texture-compression-benchmark LANGUAGES CXX)

qt_add_executable(texture_compression_benchmark
    main.cpp
    BenchmarkItem.h
    BenchmarkItem.cpp
)

qt_add_qml_module(texture_compression_benchmark
    URI TextureCompressionBenchmark
    VERSION 1.0
    RESOURCE_PREFIX /qt/qml
    QML_FILES Main.qml
)

qt_add_resources(texture_compression_benchmark "benchmark_data"
    PREFIX "/benchmark"
    BASE "${CMAKE_SOURCE_DIR}/unittests/nucleus/data/quad"
    FILES "${CMAKE_SOURCE_DIR}/unittests/nucleus/data/quad/merged.jpg"
)

qt_add_resources(texture_compression_benchmark "fonts"
    BASE "${alpineapp_fonts_SOURCE_DIR}"
    PREFIX "/fonts"
    FILES "${alpineapp_fonts_SOURCE_DIR}/Roboto/Roboto-Regular.ttf"
)

target_link_libraries(texture_compression_benchmark PUBLIC gl_engine Qt::Quick Qt::QuickControls2)
alp_configure_target(texture_compression_benchmark)

if (ANDROID)
    set_target_properties(texture_compression_benchmark PROPERTIES
        QT_ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android"
        QT_ANDROID_PACKAGE_NAME "org.alpinemaps.texturecompressionbenchmark"
        QT_ANDROID_VERSION_NAME "1.0"
        QT_ANDROID_VERSION_CODE 1
    )
    install(TARGETS texture_compression_benchmark
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (EMSCRIPTEN)
    install(
        FILES
            "$<TARGET_FILE_DIR:texture_compression_benchmark>/texture_compression_benchmark.js"
            "$<TARGET_FILE_DIR:texture_compression_benchmark>/texture_compression_benchmark.wasm"
            "$<TARGET_FILE_DIR:texture_compression_benchmark>/texture_compression_benchmark.html"
            "$<TARGET_FILE_DIR:texture_compression_benchmark>/qtloader.js"
        DESTINATION "${ALP_WWW_INSTALL_DIR}/texture_compression_benchmark"
    )
    install(
        FILES "$<TARGET_FILE_DIR:texture_compression_benchmark>/texture_compression_benchmark.worker.js"
        DESTINATION "${ALP_WWW_INSTALL_DIR}/texture_compression_benchmark"
        OPTIONAL
    )
endif()
