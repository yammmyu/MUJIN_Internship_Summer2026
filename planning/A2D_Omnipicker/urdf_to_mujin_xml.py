#!/usr/bin/env python3
"""
Convert a URDF file to Mujin OpenRAVE-style kinbody XML.

Simplified conventions used:
  - Each <body> stores its zero-pose world-frame pose (translation + rotationaxis).
  - STL meshes stay in link-local frame, so <geom> has no compensating offset.
    (visual/collision <origin> in URDF, if non-zero, is applied to <geom>.)
  - Inertial info is dropped.
  - Visual and collision share one mesh; collision mesh takes precedence.
  - URDF revolute/continuous -> Mujin "revolute" (limits in degrees).
  - URDF prismatic           -> Mujin "slider"   (limits in meters).
  - URDF fixed               -> Mujin "hinge" with enable="false" + zero limits.

Usage:
  python urdf_to_mujin_xml.py A2D.urdf -o A2D.kinbody.xml \
      --mesh-scale 1.0 --stl-dir meshes \
      --manipulator link_arm:gripper_center \
      --manipulator link_arm:right_gripper_center
"""

import argparse
import math
import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


# ---------- math helpers ----------

def rpy_to_R(rpy):
    """URDF roll-pitch-yaw (Rz * Ry * Rx) to 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_axis_angle_deg(R):
    """Convert 3x3 rotation matrix to (axis_xyz_tuple, angle_in_degrees)."""
    cos_a = (np.trace(R) - 1.0) / 2.0
    cos_a = max(-1.0, min(1.0, cos_a))
    angle = math.acos(cos_a)
    if angle < 1e-9:
        return (1.0, 0.0, 0.0), 0.0
    if math.pi - angle < 1e-9:
        d = np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None)
        i = int(np.argmax(d))
        axis = np.zeros(3)
        axis[i] = math.sqrt(d[i])
        if axis[i] > 1e-9:
            for j in range(3):
                if j != i:
                    axis[j] = (R[i, j] + R[j, i]) / (4.0 * axis[i])
        return tuple(float(x) for x in axis), math.degrees(angle)
    s = 2.0 * math.sin(angle)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / s
    return tuple(float(x) for x in axis), math.degrees(angle)


# ---------- URDF parsing ----------

def _parse_vec(s, default=(0.0, 0.0, 0.0)):
    if s is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in s.split()], dtype=float)


def _sanitize_axis(v):
    """OpenRAVE asserts on zero-length axes; fixed joints in URDF sometimes have axis=0 0 0."""
    if float(np.linalg.norm(v)) < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return v


def parse_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()

    links = {}
    for link in root.findall('link'):
        name = link.get('name')
        mesh = None
        for tag in ('collision', 'visual'):
            elem = link.find(tag)
            if elem is None:
                continue
            mesh_elem = elem.find('geometry/mesh')
            if mesh_elem is None:
                continue
            origin = elem.find('origin')
            mesh = {
                'filename': mesh_elem.get('filename'),
                'origin_xyz': _parse_vec(origin.get('xyz') if origin is not None else None),
                'origin_rpy': _parse_vec(origin.get('rpy') if origin is not None else None),
            }
            break
        links[name] = {'name': name, 'mesh': mesh}

    joints = []
    for jt in root.findall('joint'):
        origin = jt.find('origin')
        axis_elem = jt.find('axis')
        limit = jt.find('limit')
        parent = jt.find('parent').get('link')
        child = jt.find('child').get('link')
        joints.append({
            'name': jt.get('name'),
            'type': jt.get('type'),
            'parent': parent,
            'child': child,
            'xyz': _parse_vec(origin.get('xyz') if origin is not None else None),
            'rpy': _parse_vec(origin.get('rpy') if origin is not None else None),
            'axis': _sanitize_axis(_parse_vec(axis_elem.get('xyz') if axis_elem is not None else None,
                                                default=(1.0, 0.0, 0.0))),
            'lower': float(limit.get('lower', '0')) if limit is not None and limit.get('lower') is not None else None,
            'upper': float(limit.get('upper', '0')) if limit is not None and limit.get('upper') is not None else None,
            'velocity': float(limit.get('velocity', '0')) if limit is not None and limit.get('velocity') is not None else None,
        })

    child_set = {j['child'] for j in joints}
    roots = [n for n in links if n not in child_set]
    if len(roots) != 1:
        raise ValueError(f'Expected exactly one root link, found: {roots}')
    return links, joints, roots[0], root.get('name', 'robot')


def compute_fk(joints, root_name):
    """Forward kinematics at zero pose -> {link_name: (world_t, world_R)}."""
    pose = {root_name: (np.zeros(3), np.eye(3))}
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j['parent'], []).append(j)
    queue = [root_name]
    while queue:
        parent = queue.pop(0)
        for j in by_parent.get(parent, []):
            p_t, p_R = pose[parent]
            child_R = p_R @ rpy_to_R(j['rpy'])
            child_t = p_t + p_R @ j['xyz']
            pose[j['child']] = (child_t, child_R)
            queue.append(j['child'])
    return pose


# ---------- emission ----------

URDF_TO_MUJIN_TYPE = {
    'revolute':   'revolute',
    'continuous': 'revolute',
    'prismatic':  'slider',
    'fixed':      'hinge',
}


def _fmt(v, prec=6):
    out = f'{v:.{prec}g}'
    return '0' if out in ('-0', '-0.0') else out


def _mesh_rel(filename, stl_dir):
    if not filename:
        return None
    m = re.search(r'(?:^|/)meshes/(.+)$', filename)
    rel = m.group(1) if m else os.path.basename(filename)
    return f'{stl_dir}/{rel}'


def _emit_body(out, link, t, R, mesh_scale, stl_dir, indent='\t\t'):
    name = link['name']
    out.append(f'{indent}<body name="{name}">')
    out.append(f'{indent}\t<translation>{_fmt(t[0])} {_fmt(t[1])} {_fmt(t[2])}</translation>')
    axis, ang = R_to_axis_angle_deg(R)
    if abs(ang) > 1e-6:
        out.append(f'{indent}\t<rotationaxis>{_fmt(axis[0])} {_fmt(axis[1])} {_fmt(axis[2])} {_fmt(ang)}</rotationaxis>')
    mesh = link.get('mesh')
    if mesh and mesh['filename']:
        rel = _mesh_rel(mesh['filename'], stl_dir)
        out.append(f'{indent}\t<geom type="trimesh" name="{name}">')
        out.append(f'{indent}\t\t<collision>{rel} {mesh_scale}</collision>')
        if np.linalg.norm(mesh['origin_rpy']) > 1e-9:
            ma, mang = R_to_axis_angle_deg(rpy_to_R(mesh['origin_rpy']))
            out.append(f'{indent}\t\t<rotationaxis>{_fmt(ma[0])} {_fmt(ma[1])} {_fmt(ma[2])} {_fmt(mang)}</rotationaxis>')
        if np.linalg.norm(mesh['origin_xyz']) > 1e-9:
            ox = mesh['origin_xyz']
            out.append(f'{indent}\t\t<translation>{_fmt(ox[0])} {_fmt(ox[1])} {_fmt(ox[2])}</translation>')
        out.append(f'{indent}\t</geom>')
    out.append(f'{indent}</body>')


def _emit_joint(out, j, indent='\t\t'):
    name = j['name']
    jt = j['type']
    mtype = URDF_TO_MUJIN_TYPE.get(jt, 'hinge')
    enable = 'false' if jt == 'fixed' else 'true'
    out.append(f'{indent}<joint type="{mtype}" enable="{enable}" name="{name}">')
    out.append(f'{indent}\t<body>{j["parent"]}</body>')
    out.append(f'{indent}\t<body>{j["child"]}</body>')
    out.append(f'{indent}\t<offsetfrom>{j["child"]}</offsetfrom>')
    ax = j['axis']
    out.append(f'{indent}\t<axis>{_fmt(ax[0])} {_fmt(ax[1])} {_fmt(ax[2])}</axis>')
    if jt == 'fixed':
        out.append(f'{indent}\t<limitsdeg>0 0</limitsdeg>')
    elif jt == 'prismatic':
        if j['lower'] is not None:
            out.append(f'{indent}\t<limits>{_fmt(j["lower"])} {_fmt(j["upper"])}</limits>')
        if j['velocity']:
            out.append(f'{indent}\t<maxvel>{_fmt(j["velocity"])}</maxvel>')
    else:
        if j['lower'] is not None:
            out.append(f'{indent}\t<limitsdeg>{_fmt(math.degrees(j["lower"]))} {_fmt(math.degrees(j["upper"]))}</limitsdeg>')
        if j['velocity']:
            out.append(f'{indent}\t<maxveldeg>{_fmt(math.degrees(j["velocity"]))}</maxveldeg>')
    out.append(f'{indent}</joint>')


def convert(urdf_path, output_path=None, mesh_scale='1.0', stl_dir='meshes',
            manipulators=None, robot_name=None):
    links, joints, root_name, urdf_robot_name = parse_urdf(urdf_path)
    pose = compute_fk(joints, root_name)
    name = robot_name or urdf_robot_name

    # BFS order so parents precede children
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j['parent'], []).append(j)
    ordered, queue, seen = [root_name], [root_name], {root_name}
    while queue:
        cur = queue.pop(0)
        for j in by_parent.get(cur, []):
            if j['child'] not in seen:
                ordered.append(j['child'])
                seen.add(j['child'])
                queue.append(j['child'])

    out = [f'<robot name="{name}">', '\t<kinbody>',
           '\t\t<!-- Bodies (zero-pose world-frame poses) -->']
    for n in ordered:
        _emit_body(out, links[n], *pose[n], mesh_scale=mesh_scale, stl_dir=stl_dir)
    out.append('')
    out.append('\t\t<!-- Joints -->')
    for j in joints:
        _emit_joint(out, j)
    out.append('\t</kinbody>')

    for spec in (manipulators or []):
        if ':' not in spec:
            raise ValueError(f'manipulator must be "base:effector", got {spec!r}')
        base, effector = spec.split(':', 1)
        out += ['', f'\t<manipulator name="{effector}">',
                f'\t\t<base>{base}</base>',
                f'\t\t<effector>{effector}</effector>',
                '\t</manipulator>']
    out.append('</robot>')

    text = '\n'.join(out) + '\n'
    if output_path:
        Path(output_path).write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('urdf')
    ap.add_argument('-o', '--output', help='output XML path (default: <urdf-stem>.kinbody.xml)')
    ap.add_argument('--mesh-scale', default='1.0',
                    help='scale string appended to <collision> (e.g. 0.001 if STL is mm)')
    ap.add_argument('--stl-dir', default='meshes',
                    help='relative prefix for STL paths in output (default: meshes)')
    ap.add_argument('--manipulator', action='append', default=[],
                    help='add manipulator as "base_link:effector_link" (repeatable)')
    ap.add_argument('--name', help='override robot name (default: URDF robot name)')
    args = ap.parse_args()

    out_path = args.output or str(Path(args.urdf).with_suffix('.kinbody.xml'))
    convert(args.urdf, out_path, mesh_scale=args.mesh_scale, stl_dir=args.stl_dir,
            manipulators=args.manipulator, robot_name=args.name)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
