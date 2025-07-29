# Auto-Proofread Neuroglancer State Generator

## Overview

**Problem:** When developing and iterating on auto-proofread neurons in NEURD, frequent reprocessing makes it laborious to inspect changes using standalone Python tools. Visualizing meshes and skeletons in isolation does not provide context within the EM volume or nearby neurites.

**Solution:** This utility dynamically generates a Neuroglancer state—including EM imagery, segmentation, proofread meshes, and skeletons—for one or more segment IDs. It fetches proofread meshes and skeletons, uploads them to S3, creates the necessary `info` JSON, and then prints a sharable Neuroglancer link.

## Key Functionalities

* **Configuration Loading:** Read settings (dataset, AWS, S3 bucket/path, segment IDs, viewer dimensions) from `config.yaml`.
* **Mesh Fetching & Upload:** Retrieve proofread meshes via the NEURD API, convert to CloudVolume fragments, and upload under `s3://{bucket}/{s3_base_path}/mesh/{segment_id}:0`.
* **Info File Generation:** Create and upload the multiscale `info` JSON to `s3://{bucket}/{s3_base_path}/info` for mesh rendering.
* **Skeleton Fetching & Transformation:**
  * Fetch raw proofread skeletons as point arrays.
  * Reshape to an `[N,3]` vertex list.
  * Use `numpy.unique` to deduplicate vertices and reconstruct edge indices.
* **Skeleton Upload:** Use `nglui.skeletons.SkeletonManager` to upload vertices, edges, and radius attributes to the same S3 path.
* **Viewer Link Construction:** Build a Neuroglancer URL via `nglui.statebuilder` and `CAVEclient`, combining EM image, flat segmentation, proofread meshes, and skeleton layers.

## Installation

1. **Initialize UV**:

   ```bash
   uv init
   ```

2. **Synchronize** your environment to match `pyproject.toml`:

   ```bash
   uv sync
   ```
   
### Prerequisites

* **Python:** 3.8 or later
* **AWS Credentials:** Ensure `~/.aws/credentials` or environment variables are configured for S3 access.
* **NGLui:** v4 (Note this only supports the latest version of NGLui, v4)

## Usage

```bash
python generate_ng_state.py --config config.yaml
```

This will:

1. Fetch and upload proofread meshes.
2. Fetch, transform, and upload skeletons.
3. Write the Neuroglancer `info` JSON.
4. Print a shortened Neuroglancer viewer URL for inspection.
