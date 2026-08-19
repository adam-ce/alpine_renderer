#############################################################################
# AlpineMaps.org
# Copyright (C) 2026 Adam Celarek
# SPDX-License-Identifier: GPL-3.0-or-later
#############################################################################

project(texture-compression-benchmark LANGUAGES CXX)

alp_add_git_repository(basisu
    URL https://github.com/BinomialLLC/basis_universal.git
    COMMITISH 9bebe16726b3a61c8c213eeee3b7cffb462ef34e
    DO_NOT_ADD_SUBPROJECT)
set(BASISU_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BASISU_SSE OFF CACHE BOOL "" FORCE)
set(BASISU_ZSTD OFF CACHE BOOL "" FORCE)
set(BASISU_OPENCL OFF CACHE BOOL "" FORCE)
set(BASISU_DISABLE_ANDROID_ASTC_DECOMP ON CACHE BOOL "" FORCE)
add_subdirectory("${basisu_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/basisu" EXCLUDE_FROM_ALL)

add_library(alp_basisu_texture_compression STATIC
    ${CMAKE_SOURCE_DIR}/nucleus/utils/BasisUniversalTextureCompression.h
    ${CMAKE_SOURCE_DIR}/nucleus/utils/BasisUniversalTextureCompression.cpp
)
target_include_directories(alp_basisu_texture_compression SYSTEM PRIVATE "${basisu_SOURCE_DIR}")
target_compile_definitions(alp_basisu_texture_compression PRIVATE
    BASISD_SUPPORT_KTX2_ZSTD=0
    BASISU_SUPPORT_SSE=0
)
target_link_libraries(alp_basisu_texture_compression PUBLIC nucleus PRIVATE basisu_encoder)
if (UNIX AND NOT ANDROID AND NOT EMSCRIPTEN)
    target_link_libraries(alp_basisu_texture_compression PRIVATE m pthread)
endif()
alp_configure_target(alp_basisu_texture_compression)

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

qt_add_resources(texture_compression_benchmark "fonts"
    BASE "${alpineapp_fonts_SOURCE_DIR}"
    PREFIX "/fonts"
    FILES "${alpineapp_fonts_SOURCE_DIR}/Roboto/Roboto-Regular.ttf"
)

target_link_libraries(texture_compression_benchmark PUBLIC alp_basisu_texture_compression gl_engine Qt::Network Qt::Quick Qt::QuickControls2)
alp_configure_target(texture_compression_benchmark)

if (ANDROID)
    add_android_openssl_libraries(texture_compression_benchmark)
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
