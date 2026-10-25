"""Dependency-free URDF forward kinematics; distances in metres, angles in radians."""
import math
import xml.etree.ElementTree as ET


def identity():
    return [[float(i == j) for j in range(4)] for i in range(4)]


def multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def vector(text, default):
    result = [float(x) for x in text.split()] if text is not None else list(default)
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError('Expected three finite values')
    return result


def rotation(axis, angle):
    norm = math.sqrt(sum(x*x for x in axis))
    if norm == 0 or not math.isfinite(angle):
        raise ValueError('Invalid rotation axis or angle')
    x, y, z = [v/norm for v in axis]
    c, s = math.cos(angle), math.sin(angle)
    t = 1-c
    result = identity()
    rows = [[t*x*x+c, t*x*y-s*z, t*x*z+s*y],
            [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
            [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]
    for i in range(3):
        result[i][:3] = rows[i]
    return result


def origin(xyz, rpy):
    # URDF fixed-axis RPY: Rz(yaw) Ry(pitch) Rx(roll).
    result = multiply(multiply(rotation([0, 0, 1], rpy[2]),
                               rotation([0, 1, 0], rpy[1])),
                      rotation([1, 0, 0], rpy[0]))
    for i in range(3):
        result[i][3] = xyz[i]
    return result


def quaternion(matrix):
    """Unit quaternion in ROS order x,y,z,w, stable around 180 degrees."""
    m = matrix
    trace = sum(m[i][i] for i in range(3))
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = [(m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s,
             (m[1][0]-m[0][1])/s, s/4]
    else:
        i = max(range(3), key=lambda k: m[k][k])
        j, k = (i+1) % 3, (i+2) % 3
        s = 2 * math.sqrt(max(0, 1+m[i][i]-m[j][j]-m[k][k]))
        q = [0.0]*4
        q[i] = s/4
        q[j] = (m[j][i]+m[i][j])/s
        q[k] = (m[k][i]+m[i][k])/s
        q[3] = (m[k][j]-m[j][k])/s
    norm = math.sqrt(sum(v*v for v in q))
    return [v/norm for v in q]


class ForwardKinematics:
    def __init__(self, urdf_path, base_link, tip_link):
        root = ET.parse(urdf_path).getroot()
        links = {link.attrib['name'] for link in root.findall('link')}
        if base_link not in links or tip_link not in links:
            raise ValueError('Base or tip link is absent from URDF')
        parents = {}
        for joint in root.findall('joint'):
            child = joint.find('child').attrib['link']
            if child in parents:
                raise ValueError('Multiple parent joints for ' + child)
            parents[child] = joint
        self.chain = []
        cursor, visited = tip_link, set()
        while cursor != base_link:
            if cursor in visited or cursor not in parents:
                raise ValueError('No valid directed chain from base to tip')
            visited.add(cursor)
            joint = parents[cursor]
            kind = joint.attrib['type']
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError('Unsupported joint type: ' + kind)
            if joint.find('mimic') is not None:
                raise ValueError('Mimic joints require explicit support')
            node = joint.find('origin')
            xyz = vector(node.get('xyz') if node is not None else None, [0, 0, 0])
            rpy = vector(node.get('rpy') if node is not None else None, [0, 0, 0])
            axis_node = joint.find('axis')
            axis = vector(axis_node.get('xyz') if axis_node is not None else None, [1, 0, 0])
            if kind != 'fixed' and sum(v*v for v in axis) == 0:
                raise ValueError('Zero axis')
            self.chain.append((joint.attrib['name'], kind, origin(xyz, rpy), axis))
            cursor = joint.find('parent').attrib['link']
        self.chain.reverse()
        self.joint_names = [name for name, kind, _, _ in self.chain if kind != 'fixed']

    def compute(self, positions):
        result = identity()
        for name, kind, transform, axis in self.chain:
            motion = identity()
            if kind != 'fixed':
                value = float(positions[name])
                if not math.isfinite(value):
                    raise ValueError('Nonfinite joint position: ' + name)
                if kind == 'prismatic':
                    norm = math.sqrt(sum(v*v for v in axis))
                    for i in range(3):
                        motion[i][3] = axis[i]/norm * value
                else:
                    motion = rotation(axis, value)
            result = multiply(result, multiply(transform, motion))
        return result
