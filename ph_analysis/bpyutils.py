#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import bpy
from mathutils import Matrix, Vector


__author__ = "Yuji Ikeda"


def delete_all():
    for item in bpy.context.scene.objects:
        bpy.context.scene.objects.unlink(item)

    for item in bpy.data.objects:
        bpy.data.objects.remove(item)

    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)

    for item in bpy.data.materials:
        bpy.data.materials.remove(item)


def initialize():
    delete_all()
    # bpy.ops.view3d.view_persportho()
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'ORTHO'


def add_camera(location, upward=None):
    # Initially, the camera points the -z direction and the top view is +y direction.
    # bpy.ops.object.camera_add()
    if upward is None:
        upward = np.array([0.0, 0.0, 1.0])

    z2 = normalize_vector(location)
    x2 = normalize_vector(np.cross(upward, z2))
    y2 = normalize_vector(np.cross(z2, x2))
    array = np.vstack((x2, y2, z2)).T
    matrix = Matrix(array)
    print('matrix:', matrix)
    euler = matrix.to_euler()

    bpy.ops.object.camera_add(location=location, rotation=euler)
    obj = bpy.context.active_object
    obj.data.type = 'ORTHO'
    obj.data.ortho_scale = 10
    return obj


def add_lamp(location):
    theta, phi = get_angles_from_vector(location)
    rotation = ((0.0, theta, phi))
    bpy.ops.object.lamp_add(type='HEMI', location=location, rotation=rotation)
    obj = bpy.context.active_object
    return obj


def create_atom(p, subdivision=6, size=0.25):
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivision,
        size=size,
        location=p)
    obj = bpy.context.active_object
    return obj


def create_bond(p1, p2, vertices=72, radius=0.05):
    diff = p2 - p1
    veclen = np.linalg.norm(diff)
    location = (p1 + p2) * 0.5
    theta, phi = get_angles_from_vector(diff)
    rotation = (0.0, theta, phi)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=veclen,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.active_object
    # obj.rotation_mode = 'XYZ'
    # obj.rotation_euler[2] = phi
    # obj.rotation_euler[1] = theta
    return obj


def create_coil(p1, p2):
    # Create origin for screwing
    bpy.ops.object.empty_add(type='ARROWS', view_align=False)
    bpy.context.active_object.name = 'Empty'

    radius_circ = 0.10
    radius_coil = 0.25
    vec = p2 - p1
    length_coil = np.linalg.norm(vec)
    theta = np.arccos(vec[2] / length_coil)
    phi = np.arccos(vec[0] / np.linalg.norm(vec[:2]))
    if vec[1] < 0:
        phi *= -1

    print('theta, phi:', theta, phi)

    bpy.ops.mesh.primitive_circle_add(
        radius=radius_circ,
        location=(radius_coil, 0, 0),
        rotation=(np.pi * 0.5, 0, 0),  # XYZ Euler
    )
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    name = bpy.context.active_object.name
    print('screw_name:', name)

    # https://www.blender.org/api/blender_python_api_2_78_0/bpy.ops.object.html#bpy.ops.object.modifier_add
    bpy.ops.object.modifier_add(type='SCREW')
    bpy.context.object.modifiers["Screw"].axis = 'Z'
    bpy.context.object.modifiers["Screw"].object = bpy.data.objects["Empty"]
    bpy.context.object.modifiers["Screw"].angle = np.pi * 32
    bpy.context.object.modifiers["Screw"].steps = 360
    bpy.context.object.modifiers["Screw"].screw_offset = length_coil
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Screw")

    bpy.data.objects[name].select = True
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    obj = bpy.context.active_object
    obj.location = Vector((p2 + p1) * 0.5)

    obj.rotation_euler[2] = phi
    obj.rotation_euler[1] = theta

    return obj


def create_arrow1(loc_arrow, veclen, radius):
        loc_arrow = np.array(loc_arrow)

        loc_head = loc_arrow + np.array((0, 0, veclen * 0.5))
        bpy.ops.mesh.primitive_cone_add(
            vertices=72,
            radius1=radius * 2,
            depth=veclen * 0.3,
            location=loc_head,
        )
        cone_name = bpy.context.active_object.name
        print('cone_name:', cone_name)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=72,
            radius=radius,
            depth=veclen,
            location=loc_arrow,
        )
        cylinder_name = bpy.context.active_object.name
        print('cylinder name:', cylinder_name)

        bpy.data.objects[cone_name].select = True
        bpy.ops.object.join()

        arrow = bpy.data.objects[cylinder_name]
        return arrow


def create_arrow2(p1, p2, len_head=1.0):
        diff = p2 - p1
        len_arrow = np.linalg.norm(diff)
        radius = 0.1
        loc_arrow = (p1 + p2) * 0.5
        len_head = 1.0

        tmp = len_arrow - len_head * 0.5
        loc_head = p1 + np.array((0, 0, tmp))
        loc_axis = p1 + np.array((0, 0, (len_arrow - len_head) * 0.5))
        bpy.ops.mesh.primitive_cone_add(
            vertices=72,
            radius1=radius * 3,
            depth=len_head,
            location=loc_head,
        )
        cone_name = bpy.context.active_object.name
        print('cone_name:', cone_name)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=72,
            radius=radius,
            depth=len_arrow - len_head,
            location=loc_axis,
        )
        cylinder_name = bpy.context.active_object.name
        print('cylinder name:', cylinder_name)

        bpy.data.objects[cone_name].select = True
        bpy.ops.object.join()

        obj = bpy.data.objects[cylinder_name]
        bpy.context.scene.cursor_location = p1
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        obj.location = p1
        theta, phi = get_angles_from_vector(diff)
        obj.rotation_euler[2] = phi
        obj.rotation_euler[1] = theta

        return obj


def make_plane(vertices, name, color, alpha=0.5):
    n = len(vertices)
    faces = [range(n)]
    edges = [[i, i % n] for i in range(n)]  # [[0, 1], [1, 2], ..., [n, 0]]
    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(vertices, edges, faces)
    mesh_data.update()

    obj = bpy.data.objects.new(name, mesh_data)

    scene = bpy.context.scene
    scene.objects.link(obj)
    obj.select = True
    scene.objects.active = obj

    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_transparency = True
    mat.transparency_method = 'Z_TRANSPARENCY'
    mat.alpha = alpha
    obj.data.materials.append(mat)


def apply_material(names_obj, name_mat, color=None):
    mat = bpy.data.materials.new(name_mat)
    mat.diffuse_color = color
    for name in names_obj:
        bpy.data.objects[name].data.materials.append(mat)


def render(img_filename):
    bpy.context.scene.render.resolution_x = 2160
    bpy.context.scene.render.resolution_y = 2160
    bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
    bpy.context.scene.camera = bpy.data.objects['Camera']

    # Draw lines for object edges
    bpy.context.scene.render.use_freestyle = True

    bpy.context.scene.render.filepath = img_filename
    bpy.ops.render.render(write_still=True, use_viewport=True)


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def get_angles_from_vector(v):
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html
    vlen = np.linalg.norm(v)
    theta = np.arccos(v[2] / vlen)
    phi = np.arctan2(v[1], v[0])
    return theta, phi

