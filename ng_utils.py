# ng_utils.py
import importlib
import json
import boto3
import numpy as np
import yaml
from caveclient import CAVEclient
from nglui.skeletons import SkeletonManager
from connects_neuvue.utils import aws_utils as aws
from cloudvolume.mesh import Mesh
from nglui import statebuilder

def load_config(path="config.yaml"):
    with open(path, "r") as fp:
        return yaml.safe_load(fp)

def fetch_proofread_meshes(segment_ids, api_dataset, aws_secret_name=None):
    """
    Returns a dict mapping segment_id -> cloudvolume fragment bytes.
    """
    # auth
    secret = aws.get_secret(aws_secret_name) if aws_secret_name else aws.get_secret()
    api_mod = importlib.import_module(f"connects_neuvue.{api_dataset}.api")
    fetcher = api_mod.API(secret_dict=secret)

    out = {}
    for sid in segment_ids:
        decim = fetcher.fetch_segment_id_mesh(segment_id=sid)
        proof = fetcher.fetch_proofread_mesh(original_mesh=decim, segment_id=sid)
        verts = np.array(proof.vertices, dtype=np.float32)
        faces = np.array(proof.faces, dtype=np.uint32)
        cv_mesh = Mesh(verts, faces, segid=sid)
        out[sid] = cv_mesh.to_precomputed()
    return out

def fetch_proofread_skeletons(segment_ids, api_dataset, aws_secret_name=None):
    """
    Returns dict mapping segment_id -> (vertices, edges) arrays for proofread skeletons.
    """
    secret = aws.get_secret(aws_secret_name) if aws_secret_name else aws.get_secret()
    api_mod = importlib.import_module(f"connects_neuvue.{api_dataset}.api")
    fetcher = api_mod.API(secret_dict=secret)

    out = {}
    for sid in segment_ids:
        # fetch ndarray of shape (M,2,3) or flattened list of points
        raw = fetcher.fetch_proofread_skeleton(segment_id=sid)
        # ensure shape [N,3] and edges list
        all_pts = raw.reshape(-1, 3)
        verts, inv = np.unique(all_pts, axis=0, return_inverse=True)
        edges = inv.reshape(-1, 2)
        out[sid] = (verts, edges)
    return out

def init_skeleton_manager(seg_source, bucket, s3_base_path, vertex_attributes=["radius"], shader=None):
    """
    Initialize and return a SkeletonManager instance.
    """
    cloudpath = f"s3://{bucket}/{s3_base_path}/skeletons"
    skmgr = SkeletonManager(
        segmentation_source=seg_source,
        cloudpath=cloudpath,
        vertex_attributes=vertex_attributes,
        initialize_info=True,
        shader=shader
    )
    return skmgr

def upload_meshes_to_s3(fragments, bucket, s3_base_path, fragment_dir_name="mesh"):
    """
    Upload each fragment (bytes) under:
      s3://{bucket}/{s3_base_path}/{fragment_dir_name}/{segment_id}:0
    """
    s3 = boto3.client("s3")
    prefix = f"{s3_base_path}/{fragment_dir_name}"
    for sid, frag_bytes in fragments.items():
        key = f"{prefix}/{sid}:0"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=frag_bytes,
            ContentType="application/octet-stream",
        )
        print(f"→ uploaded mesh fragment for {sid}")
    return f"s3://{bucket}/{prefix}"

def upload_skeletons(skeletons, skmgr):
    """
    Upload each skeleton via the SkeletonManager.
    Returns the Neuroglancer skeleton source URL.
    """
    for sid, (verts, edges) in skeletons.items():
        skmgr.upload_skeleton(
            root_id=sid,
            vertices=verts,
            edges=edges,
            vertex_attribute_data={"radius": np.full(len(verts), 10.0, dtype=np.float32)}
        )
        print(f"→ uploaded skeleton for segment {sid}")
    # after uploads, skmgr.skeleton_source gives the URL
    return skmgr.skeleton_source

def write_info_file(bucket, s3_base_path, fragment_dir_name, cfg):
    """
    Create the “info” JSON for a multiscale mesh-only source
    and upload to s3://{bucket}/{s3_base_path}/info
    """
    info = {
        "@type": "neuroglancer_multiscale_volume",
        "type": "segmentation",
        "data_type": "uint64",
        "num_channels": cfg.get("num_channels", 1),
        "scales": [
            {
                "encoding": "raw",
                "chunk_sizes": cfg.get("chunk_sizes", [[64, 64, 64]]),
                "key": "fullres",
                "resolution": cfg.get("resolution", [8, 8, 30]),
                "size": cfg.get("size", [1, 1, 1]),
            }
        ],
        "mesh": fragment_dir_name,
    }
    data = json.dumps(info, indent=2).encode("utf-8")
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=f"{s3_base_path}/info",
        Body=data,
        ContentType="application/json",
    )
    print(f"→ uploaded info.json to s3://{bucket}/{s3_base_path}/info")


def build_viewer_link(
    image_source,
    seg_source,
    mesh_source,
    skeleton_source,
    mesh_layer_name,
    segment_ids,
    dimensions,
    cave_client_name,
    skeleton_manager=None,
    skeleton_layer_name="skeletons",
):
    """
    Returns a Neuroglancer URL or HTML link with image, segmentation, meshes, and optional skeletons.
    """
    # Choose Neuroglancer backend (e.g., spelunker)
    statebuilder.site_utils.set_default_neuroglancer_site(site_name='spelunker')

    vs = statebuilder.ViewerState(dimensions=dimensions)
    vs = (
        vs.add_image_layer(source=image_source, name="EM_image")
          .add_segmentation_layer(source=seg_source, name="flat_seg")
          .add_segmentation_layer(
              source=f"precomputed://{mesh_source}",
              name=mesh_layer_name,
              segments=segment_ids,
          )
            .add_segmentation_layer(source=f"precomputed://{skeleton_source}",
                                    name="proofread_skeletons_source",
                                    segments=segment_ids)
    )

    # If skeleton_manager provided, add skeleton layer
    if skeleton_manager is not None:
        skel_layer = skeleton_manager.to_segmentation_layer(
            name=skeleton_layer_name,
            uploaded_segments=True,
            segments_visible=True,
            shader=None
        )
        vs = vs.add_layer(skel_layer)

    # Launch in browser via CAVEclient
    client = CAVEclient(cave_client_name)
    return vs.to_browser(shorten=True, client=client, browser='safari')
