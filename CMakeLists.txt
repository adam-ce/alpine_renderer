#############################################################################
# Alpine Terrain Renderer
# Copyright (C) 2023 Adam Celarek <family name at cg tuwien ac at>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#############################################################################

cmake_minimum_required(VERSION 3.25)
project(alpine-renderer LANGUAGES CXX)

option(ALP_UNITTESTS "include unit test targets in the buildsystem" ON)
option(ALP_ENABLE_ADDRESS_SANITIZER "compiles atb with address sanitizer enabled (only debug, works only on g++ and clang)" OFF)
option(ALP_ENABLE_THREAD_SANITIZER "compiles atb with thread sanitizer enabled (only debug, works only on g++ and clang)" OFF)
option(ALP_ENABLE_ASSERTS "enable asserts (do not define NDEBUG)" ON)
option(ALP_ENABLE_TRACK_OBJECT_LIFECYCLE "enables debug cmd printout of constructors & deconstructors if implemented" OFF)
option(ALP_ENABLE_APP_SHUTDOWN_AFTER_60S "Shuts down the app after 60S, used for CI testing with asan." OFF)
option(ALP_ENABLE_LTO "Enable link time optimisation." OFF)
option(ALP_ENABLE_GL_ENGINE "Enable OpenGL/WebGL engine" ON)
option(ALP_ENABLE_AVLANCHE_WARNING_LAYER "Enables avalanche warning layer (requires Qt Gui in nucleus)" OFF)
option(ALP_ENABLE_LABELS "Enables label rendering" ON)

set(ALP_EXTERN_DIR "extern" CACHE STRING "name of the directory to store external libraries, fonts etc..")

if(ALP_ENABLE_TRACK_OBJECT_LIFECYCLE)
    add_definitions(-DALP_ENABLE_TRACK_OBJECT_LIFECYCLE)
endif()

if (EMSCRIPTEN)
    set(ALP_WWW_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "path to the install directory (for webassembly files, i.e., www directory)")
    option(ALP_ENABLE_THREADING "Puts the scheduler into an extra thread." OFF)
    option(ALP_ENABLE_DEV_TOOLS "HotReload, Renderstats, .. (increases binary size)" OFF)
elseif(ANDROID)
    option(ALP_ENABLE_THREADING "Puts the scheduler into an extra thread." ON)
    option(ALP_ENABLE_DEV_TOOLS "HotReload, Renderstats, .. (increases binary size)" OFF)
    option(ALP_ENABLE_POSITIONING "enable qt positioning (gnss / gps)" ON)
else()
    option(ALP_ENABLE_THREADING "Puts the scheduler into an extra thread." ON)
    option(ALP_ENABLE_DEV_TOOLS "HotReload, Renderstats, .. (increases binary size)" ON)
    option(ALP_ENABLE_POSITIONING "enable qt positioning (gnss / gps)" ON)
endif()


if (UNIX AND NOT EMSCRIPTEN AND NOT ANDROID)
    option(ALP_USE_LLVM_LINKER "use lld (llvm) for linking. it's parallel and much faster, but not installed by default.
        if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)" OFF)
endif()


include(cmake/AddRepo.cmake)
include(cmake/Version.cmake)

########################################### setup #################################################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if (ALP_ENABLE_ADDRESS_SANITIZER)
    message(NOTICE "building with address sanitizer enabled")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()
if (ALP_ENABLE_THREAD_SANITIZER)
    message(NOTICE "building with thread sanitizer enabled")
    message(WARN ": use the thread sanitizer supression file, e.g.: TSAN_OPTIONS=\"suppressions=thread_sanitizer_suppression.txt\" ./terrainbuilder")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")
endif()

if (ALP_USE_LLVM_LINKER)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")
endif()

########################################### dependencies #################################################
find_package(Qt6 REQUIRED COMPONENTS Core Gui OpenGL Network Quick QuickControls2 LinguistTools)
qt_standard_project_setup(REQUIRES 6.8)

alp_add_git_repository(renderer_static_data URL https://github.com/AlpineMapsOrg/renderer_static_data.git COMMITISH v23.11 DO_NOT_ADD_SUBPROJECT)
alp_add_git_repository(alpineapp_fonts URL https://github.com/AlpineMapsOrg/fonts.git COMMITISH v24.02 DO_NOT_ADD_SUBPROJECT)
alp_add_git_repository(doc URL https://github.com/AlpineMapsOrg/documentation.git COMMITISH origin/main DO_NOT_ADD_SUBPROJECT DESTINATION_PATH doc)


if (ANDROID)
    alp_add_git_repository(android_openssl URL https://github.com/KDAB/android_openssl.git COMMITISH origin/master DO_NOT_ADD_SUBPROJECT)
    include(${android_openssl_SOURCE_DIR}/android_openssl.cmake)
endif()

add_subdirectory(nucleus)
if (ALP_ENABLE_GL_ENGINE)
    if (ALP_ENABLE_DEV_TOOLS)
        find_package(Qt6 REQUIRED COMPONENTS Widgets Charts)
    endif()
    if (ALP_ENABLE_POSITIONING)
        find_package(Qt6 REQUIRED COMPONENTS Positioning)
    endif()
    add_subdirectory(gl_engine)
    add_subdirectory(plain_renderer)
    add_subdirectory(app)
endif()

if (ALP_UNITTESTS)
    add_subdirectory(unittests)
endif()
