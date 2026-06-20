#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

from setup_utils import log, fail, download


def safe_extract(tar_path, extract_to):
    extract_root = os.path.abspath(extract_to)
    try:
        with tarfile.open(tar_path, "r:gz") as archive:
            for member in archive.getmembers():
                member_path = os.path.abspath(os.path.join(extract_root, member.name))
                if not member_path.startswith(extract_root + os.sep):
                    fail(f"Unsafe path in Dawn archive: {member.name}")
            archive.extractall(extract_root)
    except Exception as exc:
        fail(f"Failed to extract {tar_path}: {exc}")


def release_assets(version):
    url = f"https://api.github.com/repos/google/dawn/releases/tags/v{version}"
    request = urllib.request.Request(url, headers={"User-Agent": "webigeo-build"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response).get("assets", [])
    except Exception as exc:
        fail(f"Failed to read Dawn release metadata for v{version}: {exc}")


def find_package_root(root):
    for current, _, files in os.walk(root):
        parts = os.path.normpath(current).split(os.sep)
        if "DawnConfig.cmake" in files and len(parts) >= 3 and parts[-2:] == ["cmake", "Dawn"]:
            return os.path.abspath(os.path.join(current, "..", "..", ".."))
    fail("DawnConfig.cmake was not found in the native Dawn package")


def find_asset(assets, suffix=None, prefix=None):
    matches = []
    for asset in assets:
        name = asset.get("name", "")
        if suffix is not None and not name.endswith(suffix):
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        matches.append(asset)
    if not matches:
        expected = f"*{suffix}" if suffix else f"{prefix}*"
        fail(f"No Dawn release asset matching {expected}")
    if len(matches) > 1:
        expected = f"*{suffix}" if suffix else f"{prefix}*"
        fail(f"Multiple Dawn release assets match {expected}")
    return matches[0]


def install_android_package(assets, version, install_dir, android_abi):
    if not android_abi:
        fail("--android-abi is required when --platform android is used")

    headers_asset = find_asset(assets, prefix="dawn-headers-")
    android_asset = find_asset(assets, prefix="dawn-android-")

    tmp_dir = tempfile.mkdtemp(prefix="fetchdawn_android_")
    headers_archive = os.path.join(tmp_dir, headers_asset["name"])
    android_archive = os.path.join(tmp_dir, android_asset["name"])

    try:
        download(headers_asset["browser_download_url"], headers_archive)
        download(android_asset["browser_download_url"], android_archive)
        safe_extract(headers_archive, tmp_dir)
        safe_extract(android_archive, tmp_dir)

        headers_root = os.path.join(tmp_dir, "dawn-headers")
        android_root = next(
            (
                os.path.join(tmp_dir, entry)
                for entry in os.listdir(tmp_dir)
                if entry.startswith("dawn-android-") and os.path.isdir(os.path.join(tmp_dir, entry))
            ),
            None,
        )
        if not os.path.isdir(headers_root):
            fail("dawn-headers directory was not found in Dawn headers package")
        if not android_root:
            fail("dawn-android directory was not found in Dawn Android package")

        abi_library = os.path.join(android_root, android_abi, "libwebgpu_dawn.a")
        if not os.path.exists(abi_library):
            fail(f"Dawn Android library was not found for ABI {android_abi}")

        if os.path.exists(install_dir):
            log(f"Removing existing Dawn Android package: {install_dir}")
            shutil.rmtree(install_dir)
        os.makedirs(os.path.dirname(install_dir), exist_ok=True)
        log(f"Installing Dawn Android package for {android_abi} to {install_dir}")
        shutil.move(headers_root, install_dir)
        os.makedirs(os.path.join(install_dir, "lib"), exist_ok=True)
        shutil.copy2(abi_library, os.path.join(install_dir, "lib", "libwebgpu_dawn.a"))
        log(f"Successfully fetched Dawn Android package v{version} for {android_abi}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch a prebuilt native Dawn package")
    parser.add_argument("--extern-dir", required=True, help="External dependencies directory")
    parser.add_argument("--dawn-version", required=True, help="Dawn version to fetch")
    parser.add_argument("--build-type", required=True, choices=["Debug", "Release"], help="Dawn package configuration")
    parser.add_argument("--platform", default="ubuntu-latest", help="Dawn release platform asset name")
    parser.add_argument("--android-abi", default="", help="Android ABI to install when --platform android is used")
    args = parser.parse_args()

    extern_dir = os.path.abspath(args.extern_dir)
    dawn_dir = os.path.join(extern_dir, "dawn")
    install_dir = os.path.join(dawn_dir, "install", args.build_type)
    assets = release_assets(args.dawn_version)

    if args.platform == "android":
        install_dir = os.path.join(install_dir, args.android_abi)
        install_android_package(assets, args.dawn_version, install_dir, args.android_abi)
        return 0

    asset_suffix = f"-{args.platform}-{args.build_type}.tar.gz"
    asset = find_asset(assets, suffix=asset_suffix)
    tmp_dir = tempfile.mkdtemp(prefix="fetchdawn_native_")
    archive_path = os.path.join(tmp_dir, asset["name"])

    try:
        download(asset["browser_download_url"], archive_path)
        safe_extract(archive_path, tmp_dir)
        package_root = find_package_root(tmp_dir)

        if os.path.exists(install_dir):
            log(f"Removing existing Dawn native package: {install_dir}")
            shutil.rmtree(install_dir)
        os.makedirs(os.path.dirname(install_dir), exist_ok=True)
        log(f"Installing Dawn native package to {install_dir}")
        shutil.move(package_root, install_dir)

        log(f"Successfully fetched native Dawn to {install_dir}")
        return 0
    except Exception as exc:
        log(f"Native Dawn fetch failed: {exc}")
        if os.path.exists(install_dir):
            shutil.rmtree(install_dir, ignore_errors=True)
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
