#############################################################################
# Alpine Terrain Renderer
# Copyright (C) 2026 Adam Celerek <family name at cg tuwien ac at>
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

project(alpine-renderer-plain_wgpu_renderer LANGUAGES CXX)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Network)

qt_add_executable(plain_wgpu_renderer
    main.cpp
    Window.h Window.cpp
)
set_target_properties(plain_wgpu_renderer PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
)
target_link_libraries(plain_wgpu_renderer PUBLIC webgpu_engine Qt::Core Qt::Gui Qt::Network)
target_include_directories(plain_wgpu_renderer PRIVATE .)

alp_configure_target(plain_wgpu_renderer)

alp_get_version(
    GIT_DIR             "${CMAKE_SOURCE_DIR}"
    VERSION_VAR         "ALP_VERSION"
    VERSION_INT_VAR     "ALP_VERSION_INTEGER"
)

set_target_properties(plain_wgpu_renderer PROPERTIES
    QT_ANDROID_PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/android
    QT_ANDROID_VERSION_NAME ${ALP_VERSION}
    QT_ANDROID_VERSION_CODE "${ALP_VERSION_INTEGER}"
)

add_android_openssl_libraries(plain_wgpu_renderer)

install(TARGETS plain_wgpu_renderer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
