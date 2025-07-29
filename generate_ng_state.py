#!/usr/bin/env python3
# generate_ng_state.py
import argparse
from ng_utils import (
    load_config,
    fetch_proofread_meshes,
    fetch_proofread_skeletons,
    init_skeleton_manager,
    upload_meshes_to_s3,
    upload_skeletons,
    write_info_file,
    build_viewer_link,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # meshes
    frags = fetch_proofread_meshes(
        cfg["segment_ids"], cfg["api_dataset"], cfg.get("aws_secret_name")
    )
    mesh_s3_url = upload_meshes_to_s3(
        frags, cfg["bucket"], cfg["s3_base_path"], cfg.get("fragment_dir_name", "mesh")
    )
    write_info_file(
        cfg["bucket"], cfg["s3_base_path"], cfg.get("fragment_dir_name", "mesh"), cfg
    )

    #skeletons
    skmgr = init_skeleton_manager(
        cfg["segmentation_layer_source"],
        cfg["bucket"],
        cfg["s3_base_path"]
    )
    skeletons = fetch_proofread_skeletons(cfg["segment_ids"], cfg["api_dataset"], cfg.get("aws_secret_name"))
    skeleton_source = upload_skeletons(skeletons, skmgr)

    # build and print the link (pass skmgr to include skeletons)
    url = build_viewer_link(
        cfg["image_layer_source"],
        cfg["segmentation_layer_source"],
        mesh_s3_url,
        skeleton_source,
        cfg.get("mesh_layer_name", "proofread_meshes"),
        cfg["segment_ids"],
        cfg.get("viewer_dimensions", [8,8,30]),
        cfg.get("cave_client_name"),
        skeleton_manager=skmgr,
        skeleton_layer_name="proofread_skeletons",
    )

if __name__ == "__main__":
    main()