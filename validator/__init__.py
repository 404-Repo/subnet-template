import logging
from typing import Any

import pyrender
import trimesh
import numpy as np
from pyrender import RenderFlags

from neurons import protocol


def score_response(synapse: protocol.Task404) -> float:
    return 0.0


def _normalize(tri_mesh: Any):
    verts_all = None
    try:
        for key in tri_mesh.geometry.keys():
            vg = tri_mesh.geometry[key]
            vert_np = np.asarray(vg.vertices)

            if verts_all is None:
                verts_all = vert_np
            else:
                verts_all = np.vstack((verts_all, vert_np))

        minv = verts_all.min(0)
        maxv = verts_all.max(0)

        half = (minv + maxv) / 2

        scale = maxv - minv
        scale = scale.max()
        scale = 1 / scale

        for key in tri_mesh.geometry.keys():
            vg = tri_mesh.geometry[key]
            vert_np = np.asarray(vg.vertices)
            vert_np = vert_np - half
            vert_np = vert_np * scale
            np.copyto(vg.vertices, vert_np)
    except Exception:
        logging.warning("Normalize failed, reverting to the default logic")

        vert_np = np.asarray(tri_mesh.vertices)
        minv = vert_np.min(0)
        maxv = vert_np.max(0)

        half = (minv + maxv) / 2

        scale = maxv - minv
        scale = scale.max()
        scale = 1 / scale

        vert_np = vert_np - half
        vert_np = vert_np * scale
        np.copyto(tri_mesh.vertices, vert_np)


def _render_obj(file_name: str, views=4):
    fuze_trimesh = trimesh.load(file_name)
    _normalize(fuze_trimesh)
    frame = 0

    W = 512
    H = 512
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    znear = 0.01
    zfar = 100
    pc = pyrender.IntrinsicsCamera(fx=f, fy=f, cx=cx, cy=cy, znear=znear, zfar=zfar)

    ci = np.eye(3)
    ci[0, 0] = f
    ci[1, 1] = f
    ci[0, 2] = cx
    ci[1, 2] = cy

    step = 360 // views

    images = []
    for azimd in range(0, 360, step):
        if isinstance(fuze_trimesh, trimesh.Trimesh):
            scene = trimesh.Scene()
            scene.add_geometry(fuze_trimesh)
            scene = pyrender.Scene.from_trimesh_scene(scene)
        else:
            # trimeshScene = fuze_trimesh
            scene = pyrender.Scene.from_trimesh_scene(fuze_trimesh)

        camera_matrix = np.eye(4)

        dist = 1.0
        elev = 0
        azim = azimd / 180 * np.pi
        # azim = 0
        x = dist * np.cos(elev) * np.sin(azim)
        y = dist * np.sin(elev)
        z = dist * np.cos(elev) * np.cos(azim)

        camera_matrix[0, 3] = x
        camera_matrix[1, 3] = y
        camera_matrix[2, 3] = z

        camera_matrix_out = np.eye(4)
        camera_matrix_out[0, 3] = x
        camera_matrix_out[1, 3] = z
        camera_matrix_out[2, 3] = y

        camera_matrix[0, 0] = np.cos(azim)
        camera_matrix[2, 2] = np.cos(azim)
        camera_matrix[0, 2] = np.sin(azim)
        camera_matrix[2, 0] = -np.sin(azim)

        camera_matrix_out[0, 0] = np.cos(azim)
        camera_matrix_out[1, 1] = np.cos(azim)
        camera_matrix_out[0, 1] = np.sin(azim)
        camera_matrix_out[1, 0] = -np.sin(azim)

        point_l = pyrender.PointLight(color=np.ones(3), intensity=3.0)

        nc = pyrender.Node(camera=pc, matrix=camera_matrix)
        scene.add_node(nc)
        ncl = pyrender.Node(light=point_l, matrix=camera_matrix)
        scene.add_node(ncl)

        r = pyrender.OffscreenRenderer(
            viewport_width=512, viewport_height=512, point_size=1.0
        )
        flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL

        color, depth = r.render(scene, flags=flags)

        images.append(color)

        frame += 1
    return images
