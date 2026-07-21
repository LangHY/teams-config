#!/usr/bin/env python3
"""Download Docker image from mirror and save as tar for docker load"""

import json
import hashlib
import os
import subprocess
import sys
import tarfile
import io

MIRROR = "https://dockerproxy.com"
IMAGE = "nvidia/cuda"
TAG = "12.4.1-runtime-ubuntu22.04"
OUTPUT_DIR = os.path.expanduser("~/hf_offline")

def curl(url, headers=None, output_file=None):
    """Download URL with curl"""
    cmd = ["curl", "-sL", "--connect-timeout", "30", "--max-time", "600"]
    if headers:
        for k, v in headers.items():
            cmd.extend(["-H", f"{k}: {v}"])
    if output_file:
        cmd.extend(["-o", output_file, "-w", "%{http_code}"])
    else:
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=660)
        return result.stdout
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=660)
    return result.stdout

def main():
    print(f"Downloading {IMAGE}:{TAG} from {MIRROR}")
    
    # Step 1: Get manifest list
    print("1. Getting manifest list...")
    url = f"{MIRROR}/v2/{IMAGE}/manifests/{TAG}"
    raw = curl(url, {"Accept": "application/vnd.docker.distribution.manifest.list.v2+json"})
    manifest_list = json.loads(raw)
    
    # Find amd64 manifest
    amd64_digest = None
    for m in manifest_list["manifests"]:
        if m.get("platform", {}).get("architecture") == "amd64":
            amd64_digest = m["digest"]
            break
    
    if not amd64_digest:
        print("ERROR: No amd64 manifest found")
        sys.exit(1)
    print(f"   AMD64 digest: {amd64_digest[:30]}...")
    
    # Step 2: Get image manifest
    print("2. Getting image manifest...")
    url = f"{MIRROR}/v2/{IMAGE}/manifests/{amd64_digest}"
    raw = curl(url, {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    manifest = json.loads(raw)
    
    config_digest = manifest["config"]["digest"]
    layers = manifest["layers"]
    print(f"   Config: {config_digest[:30]}...")
    print(f"   Layers: {len(layers)}")
    
    # Step 3: Download config
    print("3. Downloading config...")
    url = f"{MIRROR}/v2/{IMAGE}/blobs/{config_digest}"
    config_data = curl(url)
    config_hash = config_digest.split(":")[1]
    
    # Step 4: Download layers
    layer_files = []
    for i, layer in enumerate(layers):
        digest = layer["digest"]
        size_mb = layer["size"] / (1024 * 1024)
        layer_hash = digest.split(":")[1]
        layer_path = os.path.join(OUTPUT_DIR, f"layer_{i}_{layer_hash[:12]}.tar.gz")
        
        if os.path.exists(layer_path) and os.path.getsize(layer_path) == layer["size"]:
            print(f"4.{i} Layer {i} already downloaded ({size_mb:.1f}MB)")
            layer_files.append((layer_path, layer_hash, layer["size"]))
            continue
        
        print(f"4.{i} Downloading layer {i} ({size_mb:.1f}MB)...")
        url = f"{MIRROR}/v2/{IMAGE}/blobs/{digest}"
        curl(url, output_file=layer_path)
        
        actual_size = os.path.getsize(layer_path)
        layer_files.append((layer_path, layer_hash, actual_size))
        print(f"   Downloaded: {actual_size/(1024*1024):.1f}MB")
    
    # Step 5: Create Docker image tar
    print("5. Creating Docker image tar...")
    image_name = f"{IMAGE}:{TAG}"
    tar_path = os.path.join(OUTPUT_DIR, f"nvidia-cuda-12.4.1-runtime.tar")
    
    with tarfile.open(tar_path, "w") as tar:
        # Add config
        config_filename = f"{config_hash}.json"
        config_bytes = config_data.encode("utf-8")
        info = tarfile.TarInfo(name=config_filename)
        info.size = len(config_bytes)
        tar.addfile(info, io.BytesIO(config_bytes))
        
        # Add layers (as compressed)
        layer_filenames = []
        for i, (layer_path, layer_hash, layer_size) in enumerate(layer_files):
            layer_filename = f"{layer_hash}/layer.tar"
            layer_filenames.append(layer_filename)
            tar.add(layer_path, arcname=layer_filename)
        
        # Add manifest.json
        manifest_json = [{
            "Config": config_filename,
            "RepoTags": [image_name],
            "Layers": layer_filenames
        }]
        manifest_bytes = json.dumps(manifest_json).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))
        
        # Add repositories
        repos = {IMAGE: {TAG: layer_files[-1][1] if layer_files else ""}}
        repos_bytes = json.dumps(repos).encode("utf-8")
        info = tarfile.TarInfo(name="repositories")
        info.size = len(repos_bytes)
        tar.addfile(info, io.BytesIO(repos_bytes))
    
    tar_size = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"\nDone! Image saved to: {tar_path}")
    print(f"Size: {tar_size:.1f}MB")
    print(f"\nLoad with: docker load < {tar_path}")

if __name__ == "__main__":
    main()
