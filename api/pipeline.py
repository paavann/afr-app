"""
CAD processing pipeline — STEP → DGL graph + STL mesh.

This module wraps the UV-Net `solid_to_graph` and `solid_to_rendermesh`
processing logic as importable functions so the FastAPI endpoint can call
them directly without shelling out to CLI scripts.

Dependencies:
  - occwl (OpenCascade wrapper)
  - dgl   (Deep Graph Library)
  - trimesh
"""

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


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STEP → DGL Face-Adjacency Graph (.bin)
# ═══════════════════════════════════════════════════════════════════════════════


def build_face_adjacency_graph(
    step_path: pathlib.Path,
    curv_u: int = CURV_NUM_U_SAMPLES,
    surf_u: int = SURF_NUM_U_SAMPLES,
    surf_v: int = SURF_NUM_V_SAMPLES,
) -> DGLGraph:
    """
    Parse a STEP file and construct a DGL face-adjacency graph with UV-grid
    node (face) and edge (curve) features — identical to UV-Net's
    ``process.solid_to_graph.build_graph``.

    Args:
        step_path: Absolute path to the .step / .stp file.
        curv_u:    Number of uniform samples along each edge curve.
        surf_u:    Number of U-direction samples on each face surface.
        surf_v:    Number of V-direction samples on each face surface.

    Returns:
        A DGL graph ready for UV-Net inference.

    Raises:
        ValueError: If the STEP file contains no parseable solids.
        RuntimeError: If the OCC kernel fails during B-rep traversal.
    """
    if not PIPELINE_AVAILABLE:
        logger.warning("Pipeline dependencies not found. Returning a mock DGL graph.")
        # Return a simple mock object for testing
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

    # Build networkx-based face adjacency graph via occwl
    graph = face_adjacency(solid)

    # ------ Compute 2D UV-grids per face (node features) ------
    graph_face_feat: list[np.ndarray] = []
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        points = uvgrid(face, method="point", num_u=surf_u, num_v=surf_v)
        normals = uvgrid(face, method="normal", num_u=surf_u, num_v=surf_v)
        visibility = uvgrid(
            face, method="visibility_status", num_u=surf_u, num_v=surf_v
        )
        # Mask: inside (0) or on-boundary (2) → visible
        mask = np.logical_or(visibility == 0, visibility == 2)
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat_arr = np.asarray(graph_face_feat)

    # ------ Compute 1D U-grids per edge (edge features) ------
    graph_edge_feat: list[np.ndarray] = []
    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]
        if not edge.has_curve():
            continue
        points = ugrid(edge, method="point", num_u=curv_u)
        tangents = ugrid(edge, method="tangent", num_u=curv_u)
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat_arr = np.asarray(graph_edge_feat)

    # ------ Convert to DGL graph ------
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat_arr)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat_arr)

    logger.info(
        "Built DGL graph: %d faces, %d edges from '%s'",
        dgl_graph.num_nodes(),
        dgl_graph.num_edges(),
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STEP → STL Render Mesh
# ═══════════════════════════════════════════════════════════════════════════════


def build_render_mesh(
    step_path: pathlib.Path,
    output_stl_path: pathlib.Path,
    triangle_face_tol: float = TRIANGLE_FACE_TOL,
    angle_tol_rads: float = ANGLE_TOL_RADS,
) -> pathlib.Path:
    """
    Triangulate a STEP solid and export an STL mesh file suitable for
    Three.js rendering on the frontend.

    This mirrors UV-Net's ``process.solid_to_rendermesh`` logic.

    Args:
        step_path:         Path to the input .step file.
        output_stl_path:   Where to write the resulting .stl file.
        triangle_face_tol: Mesh tolerance relative to each B-rep face.
        angle_tol_rads:    Normal/tangent angle tolerance in radians.

    Returns:
        The path to the written .stl file.

    Raises:
        ValueError: If the solid produces no triangles.
    """
    if not PIPELINE_AVAILABLE:
        logger.warning("Pipeline dependencies not found. Generating a mock STL.")
        output_stl_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a minimal valid ASCII STL file for testing
        mock_stl = "solid mock\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid mock\n"
        output_stl_path.write_text(mock_stl)
        return output_stl_path

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
    return output_stl_path


def _triangulate_with_face_mapping(
    solid,
    triangle_face_tol: float = 0.01,
    angle_tol_rads: float = 0.1,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Triangulate all faces of a solid and return vertices, triangles, and a
    per-triangle → B-rep face index mapping.
    """
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
