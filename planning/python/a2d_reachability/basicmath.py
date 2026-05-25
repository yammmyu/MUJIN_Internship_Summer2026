# -*- coding: utf-8 -*-
# Copyright (C) 2011-2015 MUJIN Inc
"""Basic matirx, quaternion, and transform routines that are independent of OpenRAVE.

All angles are assumed in radians.
Some of the functions might coincide with openrave

"""
import numpy
from numpy import arccos, arctan, arctan2, array, c_, cos, cross, dot, eye, linalg, minimum, pi, r_, sin, sqrt, tile, transpose, vstack


def NormalizeZRotation(qarray):
    """for each quaternion, find the rotation about z that minimizes the distance between the identify (1,0,0,0).
    Return the transformed the quaternions along with the angle around the z-axis eliminated.
    qarray is a Nx4 array."""
    zangles = arctan2(-qarray[:, 3], qarray[:, 0])
    sinangles = sin(zangles)
    cosangles = cos(zangles)
    return (
        c_[
            cosangles * qarray[:, 0] - sinangles * qarray[:, 3],
            cosangles * qarray[:, 1] - sinangles * qarray[:, 2],
            cosangles * qarray[:, 2] + sinangles * qarray[:, 1],
            cosangles * qarray[:, 3] + sinangles * qarray[:, 0],
        ],
        -2.0 * zangles,
    )


normalizeZRotation = NormalizeZRotation  # deprecated


def MultiplyQuat(q0, q1):
    """multiplies a Nx4 array of quaternions with a quaternion"""
    return array(
        (
            q0[0] * q1[0] - q0[1] * q1[1] - q0[2] * q1[2] - q0[3] * q1[3],
            q0[0] * q1[1] + q0[1] * q1[0] + q0[2] * q1[3] - q0[3] * q1[2],
            q0[0] * q1[2] + q0[2] * q1[0] + q0[3] * q1[1] - q0[1] * q1[3],
            q0[0] * q1[3] + q0[3] * q1[0] + q0[1] * q1[2] - q0[2] * q1[1],
        )
    )


MultQuat = MultiplyQuat  # deprecated


def MultQuatArrayQuat(qarray, q):
    """multiplies a Nx4 array of quaternions with a quaternion"""
    return c_[
        (
            qarray[:, 0] * q[0] - qarray[:, 1] * q[1] - qarray[:, 2] * q[2] - qarray[:, 3] * q[3],
            qarray[:, 0] * q[1] + qarray[:, 1] * q[0] + qarray[:, 2] * q[3] - qarray[:, 3] * q[2],
            qarray[:, 0] * q[2] + qarray[:, 2] * q[0] + qarray[:, 3] * q[1] - qarray[:, 1] * q[3],
            qarray[:, 0] * q[3] + qarray[:, 3] * q[0] + qarray[:, 1] * q[2] - qarray[:, 2] * q[1],
        )
    ]


quatArrayTMult = MultQuatArrayQuat  # deprecated


def MultQuatQuatArray(q, qarray):
    """multiplies a quaternion q with each quaternion in the Nx4 array qarray"""
    return c_[
        (
            q[0] * qarray[:, 0] - q[1] * qarray[:, 1] - q[2] * qarray[:, 2] - q[3] * qarray[:, 3],
            q[0] * qarray[:, 1] + q[1] * qarray[:, 0] + q[2] * qarray[:, 3] - q[3] * qarray[:, 2],
            q[0] * qarray[:, 2] + q[2] * qarray[:, 0] + q[3] * qarray[:, 1] - q[1] * qarray[:, 3],
            q[0] * qarray[:, 3] + q[3] * qarray[:, 0] + q[1] * qarray[:, 2] - q[2] * qarray[:, 1],
        )
    ]


quatMultArrayT = MultQuatQuatArray  # deprecated


def MultiplyQuatArrays(qarray0, qarray1):
    """multiplies Nx4 array of quaternions with corresponding quaterion in Nx4 array. Returns a Nx4"""
    return c_[
        qarray0[:, 0] * qarray1[:, 0] - qarray0[:, 1] * qarray1[:, 1] - qarray0[:, 2] * qarray1[:, 2] - qarray0[:, 3] * qarray1[:, 3],
        qarray0[:, 0] * qarray1[:, 1] + qarray0[:, 1] * qarray1[:, 0] + qarray0[:, 2] * qarray1[:, 3] - qarray0[:, 3] * qarray1[:, 2],
        qarray0[:, 0] * qarray1[:, 2] + qarray0[:, 2] * qarray1[:, 0] + qarray0[:, 3] * qarray1[:, 1] - qarray0[:, 1] * qarray1[:, 3],
        qarray0[:, 0] * qarray1[:, 3] + qarray0[:, 3] * qarray1[:, 0] + qarray0[:, 1] * qarray1[:, 2] - qarray0[:, 2] * qarray1[:, 1],
    ]


def quatArrayRotate(qarray, trans):
    """rotates a point by an array of 4xN quaternions. Returns a 3xN vector"""
    xx = qarray[1, :] * qarray[1, :]
    xy = qarray[1, :] * qarray[2, :]
    xz = qarray[1, :] * qarray[3, :]
    xw = qarray[1, :] * qarray[0, :]
    yy = qarray[2, :] * qarray[2, :]
    yz = qarray[2, :] * qarray[3, :]
    yw = qarray[2, :] * qarray[0, :]
    zz = qarray[3, :] * qarray[3, :]
    zw = qarray[3, :] * qarray[0, :]
    return 2 * vstack(
        (
            (0.5 - yy - zz) * trans[0] + (xy - zw) * trans[1] + (xz + yw) * trans[2],
            (xy + zw) * trans[0] + (0.5 - xx - zz) * trans[1] + (yz - xw) * trans[2],
            (xz - yw) * trans[0] + (yz + xw) * trans[1] + (0.5 - xx - yy) * trans[2],
        )
    )


def quatArrayTRotate(qarray, trans):
    """rotates a point by an array of Nx4 quaternions. Returns a Nx3 vector"""
    xx = qarray[:, 1] * qarray[:, 1]
    xy = qarray[:, 1] * qarray[:, 2]
    xz = qarray[:, 1] * qarray[:, 3]
    xw = qarray[:, 1] * qarray[:, 0]
    yy = qarray[:, 2] * qarray[:, 2]
    yz = qarray[:, 2] * qarray[:, 3]
    yw = qarray[:, 2] * qarray[:, 0]
    zz = qarray[:, 3] * qarray[:, 3]
    zw = qarray[:, 3] * qarray[:, 0]
    return (
        2
        * c_[
            (0.5 - yy - zz) * trans[0] + (xy - zw) * trans[1] + (xz + yw) * trans[2],
            (xy + zw) * trans[0] + (0.5 - xx - zz) * trans[1] + (yz - xw) * trans[2],
            (xz - yw) * trans[0] + (yz + xw) * trans[1] + (0.5 - xx - yy) * trans[2],
        ]
    )


def InvertQuat(q):
    return array([q[0], -q[1], -q[2], -q[3]])


quatInverse = InvertQuat


def InvertIsometricMatrix(iso):
    """
    invert an affine matrix

    Arguments:
        iso (numpy.ndarray): 4x4 transformation matrix [rotation translation; 0 0 0 1]

    Returns:
        numpy.ndarray: Inverse of iso
    """
    if hasattr(iso, "dtype"):  # if quat is a numpy element, then use its dtype
        ret = eye(4, dtype=iso.dtype)
    else:
        ret = eye(4)
    ret[:3, :3] = iso[:3, :3].T
    ret[:3, 3] = -dot(iso[:3, 3], iso[:3, :3])
    return ret


def InvertPose(pose):
    qinv = array([pose[0], -pose[1], -pose[2], -pose[3]])
    return r_[qinv, -RotateQuatPoint(qinv, pose[4:7])]


def ConvertQuatFromAxisAngle(axis, angle):
    """angle is in radians"""
    axislength = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if axislength == 0:
        return array([1.0, 0, 0, 0])
    sinangle = sin(angle * 0.5) / axislength
    return array([cos(angle * 0.5), axis[0] * sinangle, axis[1] * sinangle, axis[2] * sinangle])


quatFromAxisAngle = ConvertQuatFromAxisAngle  # deprecated


def ConvertMatrixFromQuat(quat):
    """returnx 4x4 matrix"""
    length2 = quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2
    ilength2 = 2.0 / length2
    qq1 = ilength2 * quat[1] * quat[1]
    qq2 = ilength2 * quat[2] * quat[2]
    qq3 = ilength2 * quat[3] * quat[3]
    if hasattr(quat, "dtype"):  # if quat is a numpy element, then use its dtype
        T = eye(4, dtype=quat.dtype)
    else:
        T = eye(4)
    T[0, 0] = 1 - qq2 - qq3
    T[0, 1] = ilength2 * (quat[1] * quat[2] - quat[0] * quat[3])
    T[0, 2] = ilength2 * (quat[1] * quat[3] + quat[0] * quat[2])
    T[1, 0] = ilength2 * (quat[1] * quat[2] + quat[0] * quat[3])
    T[1, 1] = 1 - qq1 - qq3
    T[1, 2] = ilength2 * (quat[2] * quat[3] - quat[0] * quat[1])
    T[2, 0] = ilength2 * (quat[1] * quat[3] - quat[0] * quat[2])
    T[2, 1] = ilength2 * (quat[2] * quat[3] + quat[0] * quat[1])
    T[2, 2] = 1 - qq1 - qq2
    return T


matrixFromQuat = ConvertMatrixFromQuat  # deprecated


def ConvertMatrixFromPose(pose):
    """
    return 4x4 transform matrix from pose
    """
    matrix = matrixFromQuat(pose[:4])
    matrix[:3, 3] = pose[4:]
    return matrix


matrixFromPose = ConvertMatrixFromPose  # deprecated


def ConvertPoseFromMatrix(matrix):
    """
    return pose from transfrom matrix
    """
    matrix = numpy.asarray(matrix)
    return numpy.concatenate((quatFromMatrix(matrix), matrix[:3, 3]))


poseFromMatrix = ConvertPoseFromMatrix  # deprecated


def matrixFromAxisAngle(axis, angle):
    """angle is in radians"""
    return matrixFromQuat(quatFromAxisAngle(axis, angle))


def axisAngleFromQuat(quat):
    """returns the axis*angle form"""
    sinang = quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]
    if sinang == 0:
        return array([0.0, 0.0, 0.0])

    if quat[0] < 0:
        _quat = -quat
    else:
        _quat = quat
    sinang = sqrt(sinang)
    f = 2.0 * arctan2(sinang, _quat[0]) / sinang
    return array([_quat[1] * f, _quat[2] * f, _quat[3] * f])


def axisAngleFromQuat2(quat):
    """returns (axis,angle) tuple"""
    sinang = quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]
    if sinang == 0:
        return array([0.0, 0.0, 0.0]), 0.0

    if quat[0] < 0:
        # could be a list
        _quat = [-quat[0], -quat[1], -quat[2], -quat[3]]
    else:
        _quat = quat
    sinang = sqrt(sinang)
    f = 1.0 / sinang
    angle = 2.0 * arctan2(sinang, _quat[0])
    return array([_quat[1] * f, _quat[2] * f, _quat[3] * f]), angle


def quatFromMatrix(T):
    """Converts the rotation of a matrix into a quaternion.

    :param T: 3x3, 3x4, 4x4 transform
    """
    tr = T[0, 0] + T[1, 1] + T[2, 2]
    rot = array([0.0, 0.0, 0.0, 0.0])
    if tr >= 0:
        rot[0] = tr + 1
        rot[1] = T[2, 1] - T[1, 2]
        rot[2] = T[0, 2] - T[2, 0]
        rot[3] = T[1, 0] - T[0, 1]
    else:
        # find the largest diagonal element and jump to the appropriate case
        if T[1, 1] > T[0, 0]:
            if T[2, 2] > T[1, 1]:
                rot[3] = (T[2, 2] - (T[0, 0] + T[1, 1])) + 1
                rot[1] = T[2, 0] + T[0, 2]
                rot[2] = T[1, 2] + T[2, 1]
                rot[0] = T[1, 0] - T[0, 1]
            else:
                rot[2] = (T[1, 1] - (T[2, 2] + T[0, 0])) + 1
                rot[3] = T[1, 2] + T[2, 1]
                rot[1] = T[0, 1] + T[1, 0]
                rot[0] = T[0, 2] - T[2, 0]
        elif T[2, 2] > T[0, 0]:
            rot[3] = (T[2, 2] - (T[0, 0] + T[1, 1])) + 1
            rot[1] = T[2, 0] + T[0, 2]
            rot[2] = T[1, 2] + T[2, 1]
            rot[0] = T[1, 0] - T[0, 1]
        else:
            rot[1] = (T[0, 0] - (T[1, 1] + T[2, 2])) + 1
            rot[2] = T[0, 1] + T[1, 0]
            rot[3] = T[2, 0] + T[0, 2]
            rot[0] = T[2, 1] - T[1, 2]
    return rot / sqrt(rot[0] ** 2 + rot[1] ** 2 + rot[2] ** 2 + rot[3] ** 2)


def axisAngleFromMatrix(T):
    return axisAngleFromQuat(quatFromMatrix(T))


def matrixFromZXY(ZXY):
    """densowave P variables in pac program"""
    return dot(
        matrixFromAxisAngle([0, 0, 1], ZXY[2]),
        dot(
            matrixFromAxisAngle([1, 0, 0], ZXY[0]),
            matrixFromAxisAngle([0, 1, 0], ZXY[1]),
        ),
    )


def matrixFromZYX(ZYX):
    """standard euler angle order"""
    return dot(
        matrixFromAxisAngle([0, 0, 1], ZYX[2]),
        dot(
            matrixFromAxisAngle([0, 1, 0], ZYX[1]),
            matrixFromAxisAngle([1, 0, 0], ZYX[0]),
        ),
    )


def ConvertQuatFromZYX(ZYX):
    """standard euler angle order. angles are in radians."""
    return MultiplyQuat(
        ConvertQuatFromAxisAngle([0, 0, 1], ZYX[2]),
        MultiplyQuat(
            ConvertQuatFromAxisAngle([0, 1, 0], ZYX[1]),
            ConvertQuatFromAxisAngle([1, 0, 0], ZYX[0]),
        ),
    )


def matrixFromMitsubishi(XYZABC):
    """raw mitsubishi melfa basic values. XYZ are in mm, ABC are in degrees"""
    f = pi / 180.0
    T = matrixFromZYX([XYZABC[3] * f, XYZABC[4] * f, XYZABC[5] * f])
    T[0:3, 3] = XYZABC[0:3]
    return T


def matrixFromVirfitRotation(ZYX):
    return matrixFromZYX(ZYX)


def zxyFromMatrix(T, epsilon=1e-10):
    """T -> Z*X*Y

    .. code-block:: python

      from sympy import *
      x,y,z = Symbol('x'), Symbol('y'), Symbol('z')
      Rx = Matrix(3,3,[1,0,0,0,cos(x),-sin(x),0,sin(x),cos(x)])
      Ry = Matrix(3,3,[cos(y),0,sin(y),0,1,0,-sin(y),0,cos(y)])
      Rz = Matrix(3,3,[cos(z),-sin(z),0,sin(z),cos(z),0,0,0,1])
      Rz*Rx*Ry

     [-sin(x)*sin(y)*sin(z) + cos(y)*cos(z), -sin(z)*cos(x),  sin(x)*sin(z)*cos(y) + sin(y)*cos(z)]
     [ sin(x)*sin(y)*cos(z) + sin(z)*cos(y),  cos(x)*cos(z), -sin(x)*cos(y)*cos(z) + sin(y)*sin(z)]
     [                       -sin(y)*cos(x),         sin(x),                         cos(x)*cos(y)]

    multiplied by sign(cos(x)), but because there's two solutions, force choosing here

    Usage
    -----
    - densowave wincaps dw3 scenegraph node transforms
    """
    if abs(T[2][0]) < 1e-10 and abs(T[2][2]) < 1e-10:
        sinx = T[2][1]
        x = pi / 2 if sinx > 0 else -pi / 2
        z = 0.0
        y = arctan2(sinx * T[1][0], T[0][0])
    else:
        y = arctan2(-T[2][0], T[2][2])
        siny = sin(y)
        cosy = cos(y)
        Ryinv = array([[cosy, 0, -siny], [0, 1, 0], [siny, 0, cosy]])
        Rzx = dot(T[0:3, 0:3], Ryinv)
        x = arctan2(Rzx[2][1], Rzx[2][2])
        z = arctan2(Rzx[1][0], Rzx[0][0])
    return array([x, y, z])


def matrixFromZYZ(ZYZ):
    """Z1 * Y * Z2"""
    return dot(
        matrixFromAxisAngle([0, 0, 1], ZYZ[0]),
        dot(
            matrixFromAxisAngle([0, 1, 0], ZYZ[1]),
            matrixFromAxisAngle([0, 0, 1], ZYZ[2]),
        ),
    )


def zyzFromMatrix(T, epsilon=1e-10):
    """T -> Z1*Y*Z2

    .. code-block:: python

      from sympy import *
      z1,y,z2 = Symbol('z1'), Symbol('y'), Symbol('z2')
      Rz1 = Matrix(3,3,[cos(z1),-sin(z1),0,sin(z1),cos(z1),0,0,0,1])
      Ry = Matrix(3,3,[cos(y),0,sin(y),0,1,0,-sin(y),0,cos(y)])
      Rz2 = Matrix(3,3,[cos(z2),-sin(z2),0,sin(z2),cos(z2),0,0,0,1])
      Rz1*Ry*Rz2

    [-sin(z1)*sin(z2) + cos(y)*cos(z1)*cos(z2), -sin(z1)*cos(z2) - sin(z2)*cos(y)*cos(z1), sin(y)*cos(z1)]
    [ sin(z1)*cos(y)*cos(z2) + sin(z2)*cos(z1), -sin(z1)*sin(z2)*cos(y) + cos(z1)*cos(z2), sin(y)*sin(z1)]
    [                          -sin(y)*cos(z2),                            sin(y)*sin(z2),         cos(y)]

    multiplied by sign(sin(y)), but because there's two solutions, force choosing here


    for iter in range(10000):
        ZYZ = 2*pi*(random.rand(3)-0.5)
        T = basicmath.matrixFromZYZ(ZYZ)
        newzyz = basicmath.zyzFromMatrix(T)
        Tnew = basicmath.matrixFromZYZ(newzyz)
        err = sum(abs(T-Tnew))
        assert(err<=1e-10)

    Usage
    -----
    - kawasaki OAT convention
    """
    if (abs(T[2][0]) < epsilon and abs(T[2][2]) < epsilon) or (abs(T[1][2]) < epsilon and abs(T[0][2]) < epsilon):
        # most likely sin(y) is 0
        cosy = T[2][2]
        y = 0 if cosy > 0 else pi  # need arccos?
        z1 = arctan2(T[1][0], T[1][1])
        z2 = 0.0
    else:
        z2 = arctan2(T[2][1], -T[2][0])
        z1 = arctan2(T[1][2], T[0][2])
        sinz2 = sin(z2)
        cosz2 = cos(z2)
        if abs(sinz2) > abs(cosz2):
            y = arctan2(T[2][1] / sinz2, T[2][2])
        else:
            y = arctan2(-T[2][0] / cosz2, T[2][2])
    return array([z1, y, z2])


def ConvertZYXFromQuat(q, epsilon=1e-10):
    return zyxFromMatrix(ConvertMatrixFromQuat(q), epsilon)


def zyxFromMatrix(T, epsilon=1e-10):
    """T -> Z*Y*X

    .. code-block:: python

      from sympy import *
      x,y,z = Symbol('x'), Symbol('y'), Symbol('z')
      Rx = Matrix(3,3,[1,0,0,0,cos(x),-sin(x),0,sin(x),cos(x)])
      Ry = Matrix(3,3,[cos(y),0,sin(y),0,1,0,-sin(y),0,cos(y)])
      Rz = Matrix(3,3,[cos(z),-sin(z),0,sin(z),cos(z),0,0,0,1])
      Rz*Ry*Rx

    [cos(y)*cos(z), -cos(x)*sin(z) + cos(z)*sin(x)*sin(y),  sin(x)*sin(z) + cos(x)*cos(z)*sin(y)]
    [cos(y)*sin(z),  cos(x)*cos(z) + sin(x)*sin(y)*sin(z), -cos(z)*sin(x) + cos(x)*sin(y)*sin(z)]
    [      -sin(y),                         cos(y)*sin(x),                         cos(x)*cos(y)]


    multiplied by sign(cos(x)), but because there's two solutions, force choosing here

    Usage
    -----
    - densowave P variables in pac program
    - mitsubishi ABC
    - virfit

    """
    if abs(T[2][1]) < epsilon and abs(T[2][2]) < epsilon:
        y = pi / 2 if T[2, 0] <= 0 else -pi / 2
        if y > 0:
            xminusz = arctan2(T[0, 1], T[1, 1])
            x = xminusz
            z = 0
        else:
            xplusz = -arctan2(T[0, 1], T[1, 1])
            x = xplusz
            z = 0
    else:
        x = arctan2(T[2, 1], T[2, 2])
        sinx = sin(x)
        cosx = cos(x)
        Rxinv = array([[1, 0, 0], [0, cosx, sinx], [0, -sinx, cosx]])
        Rzy = dot(T[0:3, 0:3], Rxinv)
        y = arctan2(-Rzy[2, 0], Rzy[2, 2])
        z = arctan2(-Rzy[0, 1], Rzy[1, 1])
    return array([x, y, z])


def RotateQuatPoint(q, trans):
    """rotates a point by a 4-elt quaternion. Returns a 3 elt vector"""
    xx = q[1] * q[1]
    xy = q[1] * q[2]
    xz = q[1] * q[3]
    xw = q[1] * q[0]
    yy = q[2] * q[2]
    yz = q[2] * q[3]
    yw = q[2] * q[0]
    zz = q[3] * q[3]
    zw = q[3] * q[0]
    return 2 * array(
        (
            (0.5 - yy - zz) * trans[0] + (xy - zw) * trans[1] + (xz + yw) * trans[2],
            (xy + zw) * trans[0] + (0.5 - xx - zz) * trans[1] + (yz - xw) * trans[2],
            (xz - yw) * trans[0] + (yz + xw) * trans[1] + (0.5 - xx - yy) * trans[2],
        )
    )


quatRotate = RotateQuatPoint  # deprecated


def RotateQuatPoints(q, transarray):
    """rotates a set of points in Nx3 transarray by a quaternion. Returns a Nx3 vector"""
    xx = q[1] * q[1]
    xy = q[1] * q[2]
    xz = q[1] * q[3]
    xw = q[1] * q[0]
    yy = q[2] * q[2]
    yz = q[2] * q[3]
    yw = q[2] * q[0]
    zz = q[3] * q[3]
    zw = q[3] * q[0]
    return (
        2
        * c_[
            (
                (0.5 - yy - zz) * transarray[:, 0] + (xy - zw) * transarray[:, 1] + (xz + yw) * transarray[:, 2],
                (xy + zw) * transarray[:, 0] + (0.5 - xx - zz) * transarray[:, 1] + (yz - xw) * transarray[:, 2],
                (xz - yw) * transarray[:, 0] + (yz + xw) * transarray[:, 1] + (0.5 - xx - yy) * transarray[:, 2],
            )
        ]
    )


quatRotateArrayT = RotateQuatPoints  # deprecated


def CreateQuatRotateDirection(sourcedir, targetdir):
    """Return the minimal quaternion that orients sourcedir to targetdir

    :param sourcedir: direction of the original vector, 3 values
    :param targetdir: new direction, 3 values
    """
    rottodirection = cross(sourcedir, targetdir)
    fsin = linalg.norm(rottodirection)
    fcos = dot(sourcedir, targetdir)
    if fsin > 0:
        return ConvertQuatFromAxisAngle(rottodirection * (1 / fsin), arctan2(fsin, fcos))

    if fcos < 0:
        # hand is flipped 180, rotate around x axis
        rottodirection = array((1.0, 0, 0))
        rottodirection -= sourcedir * dot(sourcedir, rottodirection)
        if sum(rottodirection**2) < 1e-8:
            rottodirection = array((0, 0, 1.0))
            rottodirection -= sourcedir * dot(sourcedir, rottodirection)

        rottodirection /= linalg.norm(rottodirection)
        return ConvertQuatFromAxisAngle(rottodirection, arctan2(fsin, fcos))

    return array((1.0, 0, 0, 0))


def MultiplyPose(pose0, pose1):
    """multiplies two poses first 4 elements are quaternion, last 3 are translation"""
    return r_[
        MultiplyQuat(pose0[0:4], pose1[0:4]),
        array(pose0[4:7]) + RotateQuatPoint(pose0[0:4], pose1[4:7]),
    ]


def poseMultArrayT(pose, posearray):
    """multiplies a pose with an array of poses (each pose is a quaterion + translation)"""
    return c_[
        quatMultArrayT(pose[0:4], posearray[:, 0:4]),
        quatRotateArrayT(pose[0:4], posearray[:, 4:7]) + tile(pose[4:7], (len(posearray), 1)),
    ]


def quatArrayTDist(q, qarray):
    """computes the natural distance (Haar measure) for quaternions, q is a 4-element array, qarray is Nx4"""
    return arccos(minimum(1.0, numpy.abs(dot(qarray, q))))


def TransformPoints(T, points):
    """Transforms a Nxk array of points by an affine matrix"""
    return dot(points, transpose(T[0:-1, 0:-1])) + T[0:-1, -1].T  # don't need tile since numpy will do that automatically


def TransformNormals(T, normals):
    """Transforms a Nxk array of points by an affine matrix (k+1) x (k+1)"""
    return dot(normals, transpose(T[:-1, :-1]))


def TransformPosePoints(pose, points):
    """transforms a list of points by pose

    :param pose: 7-element pose quaterion + translation
    """
    # TODO should optimize this
    # for now, just do whatever works
    T = ConvertMatrixFromQuat(pose)
    T[:3, 3] = pose[4:7]
    return TransformPoints(T, points)


def TransformPlane(transform, plane):
    """
    dot(newPlane^T, dot(transform, point)) = dot(plane^T, point) <=>
    dot(newPlane^T, transform) = plane1^T <=>
    dot(transform^T, newPlane) = plane1 <=>
    newPlane = dot(inv(transform^T), plane1)
    """
    return dot(InvertIsometricMatrix(transform).T, plane)


transformPoints = TransformPoints  # deprecated


def TransformInversePoints(T, points):
    """Transforms a Nxk array of points by the inverse of an affine matrix"""
    kminus = T.shape[1] - 1
    return dot(points - tile(T[0:kminus, kminus], (len(points), 1)), T[0:kminus, 0:kminus])


transformInversePoints = TransformInversePoints  # deprecated


def ComputePoseDistSqr(pose0, pose1, quatweight=1.0):
    """computes the squared distance between two poses"""
    diff2 = numpy.abs(pose0 - pose1) ** 2
    qdist0 = numpy.sum(diff2[0:4])
    qdist1 = numpy.sum((pose0[0:4] + pose1[0:4]) ** 2)
    return min(qdist0, qdist1) * quatweight + numpy.sum(diff2[4:7])


def ComputePoseArrayDistSqr(pose0, posearray, quatweight=1.0):
    """computes the squared distance between pose0 and all poses in posearray"""
    pose0tiled = tile(pose0, (len(posearray), 1))
    diff2 = numpy.abs(pose0tiled - posearray) ** 2
    qdists0 = numpy.sum(diff2[:, 0:4], 1)
    qdists1 = numpy.sum((pose0tiled[:, 0:4] + posearray[:, 0:4]) ** 2, 1)
    return minimum(qdists0, qdists1) * quatweight + numpy.sum(diff2[:, 4:7], 1)


def ComputeTransformLookat(lookat, camerapos, cameraup):
    """Returns a camera 4x4 matrix (array) that looks along a ray with a desired up vector.

    :param lookat: the point space to look at, the camera will rotation and zoom around this point
    :param campos: the position of the camera in space
    :param camup: vector from the camera
    """
    camerapos = array(camerapos)
    cameraup = array(cameraup)
    cameradir = array(lookat) - camerapos
    cameradirlen = linalg.norm(cameradir)
    if cameradirlen > 1e-15:
        cameradir *= 1 / cameradirlen
    else:
        cameradir = array([0.0, 0.0, 1.0])
    up = cameraup - cameradir * dot(cameradir, cameraup)
    cameradirlen = linalg.norm(up)
    if cameradirlen < 1e-8:
        up = array([0.0, 1.0, 0.0])
        up -= cameradir * dot(cameradir, up)
        cameradirlen = linalg.norm(up)
        if cameradirlen < 1e-8:
            up = array([1.0, 0.0, 0.0])
            up -= cameradir * dot(cameradir, up)
            cameradirlen = linalg.norm(up)
    up *= 1 / cameradirlen
    right = cross(up, cameradir)
    t = eye(4)
    t[0, 0] = right[0]
    t[0, 1] = up[0]
    t[0, 2] = cameradir[0]
    t[0, 3] = camerapos[0]
    t[1, 0] = right[1]
    t[1, 1] = up[1]
    t[1, 2] = cameradir[1]
    t[1, 3] = camerapos[1]
    t[2, 0] = right[2]
    t[2, 1] = up[2]
    t[2, 2] = cameradir[2]
    t[2, 3] = camerapos[2]
    return t


transformLookat = ComputeTransformLookat  # deprecated


def LogQuaterion(q):
    """computes the log of the quaternion
    q = cos(theta) + axis*sin(theta)
    log(q) = axis*theta
    """
    return r_[0.0, 0.5 * axisAngleFromQuat(q)]


def FindClosestQuaternionAlongSlerp(qtest, qslerp0, qslerp1):
    """finds the closest quaternion along the slerp of qslerp0 and qslerp1 to qtest

    qslerpd = qslerp0**-1 * qslerp1
    slerp(qslerp0, qslerp1, t) = qslerp0 * qslerpd**t
    dslerp/dt (qslerp0, qslerp1, t) = qslerp0 * qslerpd**t * log(qslerpd)

    dist(t) = dot(slerp(qslerp0, qslerp1, t), qtest)
    ddist/dt(t) = dot(qslerp0 * qslerpd**t * log(qslerpd), qtest) = 0

    dot(qslerpd**t, qslerp0**-1 * qtest * log(qslerpd)**-1) = 0

    qslerpd = cos(angle) + axis*sin(angle)
    qslerpd**t = cos(angle*t) + axis*sin(angle*t)

    :return: the closest quaternion on the arc and the abs dot-prodct distance to qtest
    """
    # compute distances to the current edge points
    if dot(qslerp0, qslerp1) < 0:
        qslerp1 = -array(qslerp1)
    qbestdist = abs(dot(qslerp0, qtest))
    qbesttime = 0.0
    testdist = abs(dot(qslerp1, qtest))
    if testdist > qbestdist:
        qbestdist = testdist
        qbesttime = 1.0
    qslerp0inv = InvertQuat(qslerp0)
    qslerpd = MultiplyQuat(qslerp0inv, qslerp1)
    axis, angle = axisAngleFromQuat2(qslerpd)
    angle *= 0.5
    # from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
    if abs(angle) > 1e-7:
        logqslerpd = r_[0.0, axis * angle]
        qtemp = MultiplyQuat(qslerp0inv, MultiplyQuat(qtest, -logqslerpd))
        # take the dot product between (cos(angle*t) + axis*sin(angle*t)) and qtemp and set to 0
        # tan(angle*t) = -qtemp[0]/dot(axis.qtemp[1:4])
        d = dot(axis, qtemp[1:4])
        if abs(d) > 1e-7:
            time = arctan(-qtemp[0] / d) / angle
        else:
            time = abs(pi * 0.5 / angle)
        if time >= 0 and time <= 1:
            qclosest = MultiplyQuat(qslerp0, r_[cos(angle * time), sin(angle * time) * axis])
            testdist = abs(dot(qtest, qclosest))
            if testdist > qbestdist:
                qbestdist = testdist
                qbesttime = time
    return qbesttime, qbestdist


def InterpolateQuatSlerp(qslerp0, qslerp1, t):
    qslerp0inv = InvertQuat(qslerp0)
    qslerpd = MultiplyQuat(qslerp0inv, qslerp1)
    axis, angle = axisAngleFromQuat2(qslerpd)
    qd = quatFromAxisAngle(axis, angle * t)
    return MultiplyQuat(qslerp0, qd)


def InterpolateQuatSlerp2(qslerp0, qslerp1, t):

    cosHalfTheta = qslerp0[3] * qslerp1[3] + qslerp0[0] * qslerp1[0] + qslerp0[1] * qslerp1[1] + qslerp0[2] * qslerp1[2]
    if abs(cosHalfTheta) >= 1.0:
        return qslerp0

    halfTheta = arccos(cosHalfTheta)
    sinHalfTheta = sqrt(1 - cosHalfTheta * cosHalfTheta)
    if abs(sinHalfTheta) < 1e-7:
        return [
            (qslerp0[0] * 0.5 + qslerp1[0] * 0.5),
            (qslerp0[1] * 0.5 + qslerp1[1] * 0.5),
            (qslerp0[2] * 0.5 + qslerp1[2] * 0.5),
            (qslerp0[3] * 0.5 + qslerp1[3] * 0.5),
        ]

    ratioA = sin((1 - t) * halfTheta) / sinHalfTheta
    ratioB = sin(t * halfTheta) / sinHalfTheta

    return [
        (qslerp0[3] * ratioA + qslerp1[3] * ratioB),
        (qslerp0[0] * ratioA + qslerp1[0] * ratioB),
        (qslerp0[1] * ratioA + qslerp1[1] * ratioB),
        (qslerp0[2] * ratioA + qslerp1[2] * ratioB),
    ]


def ConvertRotateDirectionToAxisAngle(sourcedir, targetdir):
    """Return the minimal quaternion that orients sourcedir to targetdir"""
    rottodirection = cross(sourcedir, targetdir)
    fsin = linalg.norm(rottodirection)
    fcos = dot(sourcedir, targetdir)
    if fsin > 0:
        return rottodirection * (arctan2(fsin, fcos) / fsin)

    if fcos < 0:
        # hand is flipped 180, rotate around x axis
        rottodirection = array([1.0, 0.0, 0.0])
        rottodirection -= sourcedir * dot(sourcedir, rottodirection)
        if sum(rottodirection**2) < 1e-8:
            rottodirection = array([0.0, 0.0, 1.0])
            rottodirection -= sourcedir * dot(sourcedir, rottodirection)

        rottodirection /= linalg.norm(rottodirection)
        return rottodirection * arctan2(fsin, fcos)

    return array([0.0, 0.0, 0.0])


unitdict = {
    "m": 1.0,
    "cm": 100.0,
    "mm": 1000.0,
    "um": 1e6,
    "nm": 1e9,
    "inch": 39.370078740157481,
    "meter": 1.0,
    "0.1mm": 10000,
}


def GetUnitConversionScale(sourceunit, targetunit):
    """return a scale so that X source * scale = Y target"""
    try:
        return unitdict[targetunit] / unitdict[sourceunit]
    except KeyError as e:
        raise ValueError("Unsupported length unit: {0}".format(e))


unitMassDict = {"kg": 1.0, "g": 1000.0, "mg": 1e6}


def GetUnitConversionScaleMass(sourceunit, targetunit):
    """return a scale so that X source * scale = Y target"""
    try:
        return unitMassDict[targetunit] / unitMassDict[sourceunit]
    except KeyError as e:
        raise ValueError("Unsupported mass unit: {0}".format(e))
