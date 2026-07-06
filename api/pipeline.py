from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

try:
    import dgl
    import numpy as np
    import torch
    import trimesh
    from occwl.compound import Compound
    from occwl.entity_mapper import EntityMapper
    from occwl.graph import face_adjacency
    from occwl.io import load_step
    from occwl.uvgrid import ugrid, uvgrid
    PIPELINE_AVAILABLE = True
except Exception as e:
    import logging
    logging.warning(f"Pipeline dependencies not available: {e}")
    PIPELINE_AVAILABLE = False

if TYPE_CHECKING:
    from dgl import DGLGraph

from config import (
    CURV_NUM_U_SAMPLES,
    SURF_NUM_U_SAMPLES,
    SURF_NUM_V_SAMPLES,
    TRIANGLE_FACE_TOL,
    ANGLE_TOL_RADS,
)

logger = logging.getLogger(__name__)





def build_face_adjacency_graph(
    step_path: pathlib.Path,
    curv_u: int = CURV_NUM_U_SAMPLES,
    surf_u: int = SURF_NUM_U_SAMPLES,
    surf_v: int = SURF_NUM_V_SAMPLES,
) -> DGLGraph:
    if not PIPELINE_AVAILABLE:
        logger.warning("Pipeline dependencies not found. Returning a mock DGL graph.")
        class MockTensor:
            def permute(self, *args): return self
            def float(self): return self

        class MockGraph:
            def __init__(self):
                self.ndata = {"x": MockTensor()}
                self.edata = {"x": MockTensor()}
            def num_nodes(self): return 10
            def num_edges(self): return 20
            def to(self, device): return self
        return MockGraph()

    solids = load_step(str(step_path))
    if not solids:
        raise ValueError(
            f"CAD kernel could not extract any solids from '{step_path.name}'. "
            "The file may be corrupted or contain unsupported geometry."
        )
    solid = solids[0]

    graph = face_adjacency(solid)

    # ---- Pass 1: gather raw per-face / per-edge geometry ------------------
    face_points: list[np.ndarray] = []
    face_normals: list[np.ndarray] = []
    face_masks: list[np.ndarray] = []
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        points = uvgrid(face, method="point", num_u=surf_u, num_v=surf_v)
        normals = uvgrid(face, method="normal", num_u=surf_u, num_v=surf_v)
        visibility = uvgrid(
            face, method="visibility_status", num_u=surf_u, num_v=surf_v
        )
        # Mask: inside (0) or on-boundary (2) → visible
        mask = np.logical_or(visibility == 0, visibility == 2).astype(points.dtype)
        face_points.append(points)
        face_normals.append(normals)
        face_masks.append(mask)

    if not face_points:
        raise ValueError(
            f"No usable faces found in '{step_path.name}'."
        )

    edge_points: list[np.ndarray] = []
    edge_tangents: list[np.ndarray] = []
    valid_edges: list[tuple] = []
    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]
        if not edge.has_curve():
            continue
        points = ugrid(edge, method="point", num_u=curv_u)
        tangents = ugrid(edge, method="tangent", num_u=curv_u)
        edge_points.append(points)
        edge_tangents.append(tangents)
        valid_edges.append(edge_idx)

    # ---- Normalize: center at the origin, fit inside a size-2 cube --------
    # UV-Net paper, Sec 3.1: "the scale of the solid is normalized into a
    # cube of size 2 and centered at origin". Absolute xyz point coordinates
    # are a direct input channel to the surface/curve encoders, so without
    # this step, real-world STEP files (arbitrary units/scale/position) land
    # on a totally different coordinate distribution than whatever your
    # checkpoint was trained on. Surface normals / curve tangents are unit
    # directions and are unaffected by translation or uniform scale, so they
    # are left untouched.
    all_pts = np.concatenate(
        [p.reshape(-1, 3) for p in face_points]
        + [p.reshape(-1, 3) for p in edge_points],
        axis=0,
    )
    bbox_min = np.nanmin(all_pts, axis=0)
    bbox_max = np.nanmax(all_pts, axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = float(np.nanmax(bbox_max - bbox_min))
    scale = 2.0 / extent if extent > 1e-9 else 1.0

    graph_face_feat = [
        np.concatenate(
            ((points - center) * scale, normals, mask),
            axis=-1,
        )
        for points, normals, mask in zip(face_points, face_normals, face_masks)
    ]
    graph_face_feat_arr = np.asarray(graph_face_feat)

    graph_edge_feat = [
        np.concatenate(((points - center) * scale, tangents), axis=-1)
        for points, tangents in zip(edge_points, edge_tangents)
    ]

    # ---- Build the DGL graph ----------------------------------------------
    # networkx face-adjacency edges are undirected, but dgl.graph() always
    # builds a DIRECTED graph. Without explicitly adding the reverse
    # direction, message passing only flows one arbitrary way across each
    # adjacent face pair, and roughly half the faces silently never receive
    # any neighbor signal at all during graph convolution.
    src = [e[0] for e in valid_edges]
    dst = [e[1] for e in valid_edges]
    full_src = src + dst
    full_dst = dst + src
    # Same physical B-rep edge, same features, duplicated for both directions.
    full_edge_feat = graph_edge_feat + graph_edge_feat

    dgl_graph = dgl.graph((full_src, full_dst), num_nodes=len(graph.nodes))

    ndata = torch.from_numpy(graph_face_feat_arr).float()
    if full_edge_feat:
        edata = torch.from_numpy(np.asarray(full_edge_feat)).float()
    else:
        # No curved edges at all (unusual, but keep the tensor shape sane
        # for downstream code that always expects ndata/edata['x']).
        edata = torch.zeros((0, curv_u, 6), dtype=torch.float32)

    # ---- Guard against NaN/Inf from degenerate parametric samples ---------
    # Surface poles (sphere/cone apex), degenerate trims, and near-singular
    # edges can produce NaN/Inf samples from uvgrid/ugrid. Left unchecked, a
    # single bad face's NaNs spread to its neighbors within 1-2 GNN layers
    # via message passing and can collapse every face's prediction to the
    # same (meaningless) class.
    if not torch.isfinite(ndata).all():
        logger.warning(
            "Non-finite values found in face UV-grid features for '%s' — "
            "sanitizing to 0. This usually indicates a degenerate/singular "
            "surface sample (e.g. a sphere or cone pole).",
            step_path.name,
        )
    if not torch.isfinite(edata).all():
        logger.warning(
            "Non-finite values found in edge U-grid features for '%s' — "
            "sanitizing to 0.",
            step_path.name,
        )
    dgl_graph.ndata["x"] = torch.nan_to_num(ndata, nan=0.0, posinf=0.0, neginf=0.0)
    dgl_graph.edata["x"] = torch.nan_to_num(edata, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "Built DGL graph: %d faces, %d directed edges (%d unique B-rep edges) from '%s'",
        dgl_graph.num_nodes(),
        dgl_graph.num_edges(),
        len(valid_edges),
        step_path.name,
    )
    return dgl_graph


def save_graph(dgl_graph: DGLGraph, output_path: pathlib.Path) -> pathlib.Path:
    """Persist a DGL graph to disk as a .bin file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not PIPELINE_AVAILABLE:
        output_path.write_bytes(b"mock graph data")
    else:
        dgl.data.utils.save_graphs(str(output_path), [dgl_graph])
    logger.info("Saved graph → %s", output_path)
    return output_path



def build_render_mesh(
    step_path: pathlib.Path,
    output_stl_path: pathlib.Path,
    triangle_face_tol: float = TRIANGLE_FACE_TOL,
    angle_tol_rads: float = ANGLE_TOL_RADS,
) -> pathlib.Path:
    if not PIPELINE_AVAILABLE:
        logger.warning("Pipeline dependencies not found. Generating a mock STL.")
        output_stl_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a minimal valid ASCII STL file for testing
        mock_stl = "solid mock\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid mock\n"
        output_stl_path.write_text(mock_stl)
        return output_stl_path, None

    solid = Compound.load_from_step(str(step_path))
    verts, tris, tri_mapping = _triangulate_with_face_mapping(
        solid, triangle_face_tol, angle_tol_rads
    )

    if verts is None or tris is None:
        raise ValueError(
            f"Triangulation produced no geometry for '{step_path.name}'. "
            "The B-rep may have degenerate or empty faces."
        )

    output_stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=tris)
    mesh.export(str(output_stl_path))
    logger.info("Exported STL mesh → %s  (%d verts, %d tris)", output_stl_path, len(verts), len(tris))
    return output_stl_path, tri_mapping



def _triangulate_with_face_mapping(
    solid,
    triangle_face_tol: float = 0.01,
    angle_tol_rads: float = 0.1,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    solid.triangulate_all_faces(
        triangle_face_tol=triangle_face_tol, angle_tol_rads=angle_tol_rads
    )

    mapper = EntityMapper(solid)
    verts_list: list[np.ndarray] = []
    tris_list: list[np.ndarray] = []
    tri_mapping_list: list[np.ndarray] = []
    vert_counter = 0

    for face in solid.faces():
        face_index = mapper.face_index(face)
        face_verts, face_tris = face.get_triangles()
        if len(face_tris) == 0:
            continue
        face_tris = face_tris + vert_counter
        vert_counter += face_verts.shape[0]
        face_mapping = np.ones(face_tris.shape[0]) * face_index
        verts_list.append(face_verts)
        tris_list.append(face_tris)
        tri_mapping_list.append(face_mapping)

    if not verts_list:
        return None, None, None

    verts = np.concatenate(verts_list, axis=0).astype(np.float32)
    tris = np.concatenate(tris_list, axis=0).astype(np.int32)
    tri_mapping = np.concatenate(tri_mapping_list, axis=-1).astype(np.int32)
    return verts, tris, tri_mapping