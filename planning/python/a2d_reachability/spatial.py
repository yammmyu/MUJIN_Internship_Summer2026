import contextlib
import math
import typing
from typing import List, overload

import six

serde = None


if typing.TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Protocol, SupportsIndex, TypeVar, Union

    _T = TypeVar("_T", contravariant=True)

    class Observer(Protocol[_T]):
        def onNext(self, value):
            # type: (_T) -> None
            pass

        def onError(self, exc):
            # type: (Exception) -> None
            pass

        def onCompleted(self):
            # type: () -> None
            pass


class _FixedSizeObservableList(List[float]):
    def __init__(self, __iterable):
        # type: (Iterable[float]) -> None
        super(_FixedSizeObservableList, self).__init__(__iterable)
        self._observers = []  # type: list[Observer[list[float]]]

    def __add__(self, other):  # type: ignore[override]
        # type: (Iterable[float]) -> _FixedSizeObservableList
        raise NotImplementedError()

    def __delitem__(self, key):
        # type: (Union[SupportsIndex, slice]) -> None
        raise NotImplementedError()

    def __delslice__(self, i, j):
        # type: (SupportsIndex, int) -> None
        raise NotImplementedError()

    def __iadd__(self, other):  # type: ignore[override]
        # type: (Iterable[float]) -> _FixedSizeObservableList
        raise NotImplementedError()

    def __imul__(self, other):  # type: ignore[override]
        # type: (int) -> _FixedSizeObservableList
        raise NotImplementedError()

    def __mul__(self, other):  # type: ignore[override]
        # type: (int) -> _FixedSizeObservableList
        raise NotImplementedError()

    def __rmul__(self, other):  # type: ignore[override]
        # type: (int) -> _FixedSizeObservableList
        raise NotImplementedError()

    def __setitem__(self, key, value):  # type: ignore[override]
        # type: (Union[int, slice], Union[float, Iterable[float]]) -> None
        if isinstance(key, slice) and not isinstance(value, float):
            value = [float(v) for v in value]
            if len(self[key]) != len(value):
                raise ValueError("Incompatible dimension of %r, expected %d" % (value, len(self[key])))
            if self[key] != value:
                super(_FixedSizeObservableList, self).__setitem__(key, value)
                for obs in self._observers:
                    obs.onNext(self)
        elif isinstance(key, int) and isinstance(value, float):
            value = float(value)
            if self[key] != value:
                super(_FixedSizeObservableList, self).__setitem__(key, value)
                for obs in self._observers:
                    obs.onNext(self)
        else:
            raise NotImplementedError()

    def append(self, __object):
        # type: (float) -> None
        raise NotImplementedError()

    def extend(self, __iterable):
        # type: (Iterable[float]) -> None
        raise NotImplementedError()

    def insert(self, __index, __object):
        # type: (SupportsIndex, float) -> None
        raise NotImplementedError()

    def pop(self, __index=-1):
        # type: (SupportsIndex) -> float
        raise NotImplementedError()

    def remove(self, __value):
        # type: (float) -> None
        raise NotImplementedError()

    def reverse(self):
        # type: () -> None
        raise NotImplementedError()

    def sort(self, cmp=lambda _1, _2: 0, key=lambda _: 0, reverse=False):
        # type: (Callable[[float, float], Any], Callable[[float], Any], bool) -> None
        raise NotImplementedError()

    def Subscribe(self, obs):
        # type: (Observer[list[float]]) -> None
        self._observers.append(obs)


class Vec3(_FixedSizeObservableList):
    def __init__(self, t):
        # type: (Iterable[float]) -> None
        if not isinstance(t, Vec3):
            t = [float(v) for v in t]
            assert len(t) == 3, t
        super(Vec3, self).__init__(t)

    @property
    def x(self):
        # type: () -> float
        return self[0]

    @x.setter
    def x(self, value):
        # type: (float) -> None
        self[0] = float(value)

    @property
    def y(self):
        # type: () -> float
        return self[1]

    @y.setter
    def y(self, value):
        # type: (float) -> None
        self[1] = float(value)

    @property
    def z(self):
        # type: () -> float
        return self[2]

    @z.setter
    def z(self, value):
        # type: (float) -> None
        self[2] = float(value)

    def __mul__(self, other):  # type: ignore[override]
        # type: (float) -> Vec3
        a = float(other)
        return Vec3([self.x * a, self.y * a, self.z * a])

    def __rmul__(self, other):  # type: ignore[override]
        # type: (float) -> Vec3
        return self * other

    def __imul__(self, other):  # type: ignore[override]
        # type: (float) -> Vec3
        self[:] = self * other
        return self

    if six.PY2:

        def __div__(self, other):
            # type: (float) -> Vec3
            return self * (1 / float(other))

        def __idiv__(self, other):
            # type: (float) -> Vec3
            self[:] = self / other
            return self

    if six.PY3:

        def __truediv__(self, other):
            # type: (float) -> Vec3
            return self * (1 / float(other))

        def __itruediv__(self, other):
            # type: (float) -> Vec3
            self[:] = self / other
            return self

    def __neg__(self):
        # type: () -> Vec3
        return self * -1

    def __add__(self, other):  # type: ignore[override]
        # type: (Iterable[float]) -> Vec3
        other = list(other)
        if len(other) == 3:
            t = Vec3(other)
            return Vec3([self.x + t.x, self.y + t.y, self.z + t.z])
        raise TypeError("Vec3 does not add with %r" % other)

    def __iadd__(self, other):  # type: ignore[override]
        # type: (Iterable[float]) -> Vec3
        other = list(other)
        if len(other) == 3:
            self[:] = self + other
            return self
        raise TypeError("Vec3 does not add with %r" % other)

    def __sub__(self, other):
        # type: (Iterable[float]) -> Vec3
        other = list(other)
        if len(other) == 3:
            return self + (-Vec3(other))
        raise TypeError("Vec3 does not subtract with %r" % other)

    def __isub__(self, other):
        # type: (Iterable[float]) -> Vec3
        other = list(other)
        if len(other) == 3:
            self[:] = self - other
            return self
        raise TypeError("Vec3 does not subtract with %r" % other)

    def __deepcopy__(self, memodict=None):
        # type: (None) -> Vec3
        return Vec3(self[:])

    @property
    def q(self):
        # type: () -> Quaternion
        """Returns the quaternion whose rotation vector is this vector"""
        theta = math.sqrt(sum(v * v for v in self))
        if theta == 0.0:
            return Quaternion.I
        else:
            c = math.cos(theta * 0.5)
            d = math.sin(theta * 0.5)
            x, y, z = self / theta
            return Quaternion([c, x * d, y * d, z * d], normalized=True)

    def RotDeg(self, deg):
        # type: (float) -> Quaternion
        deg = float(deg)
        norm = math.sqrt(sum(v * v for v in self))
        if norm == 0.0:
            raise ValueError("Cannot rotate around null vector")
        return (self * (deg / 180.0 * math.pi / norm)).q

    I = None  # type: Vec3


Vec3.I = Vec3([0, 0, 0])
X = Vec3([1, 0, 0])
Y = Vec3([0, 1, 0])
Z = Vec3([0, 0, 1])


class Translation(_FixedSizeObservableList):
    def __init__(self, t):
        # type: (Iterable[float]) -> None
        if not isinstance(t, Translation):
            t = [float(v) for v in t]
            assert len(t) == 3, t
        super(Translation, self).__init__(t)

    @property
    def x(self):
        # type: () -> float
        return self[0]

    @x.setter
    def x(self, value):
        # type: (float) -> None
        self[0] = float(value)

    @property
    def y(self):
        # type: () -> float
        return self[1]

    @y.setter
    def y(self, value):
        # type: (float) -> None
        self[1] = float(value)

    @property
    def z(self):
        # type: () -> float
        return self[2]

    @z.setter
    def z(self, value):
        # type: (float) -> None
        self[2] = float(value)

    def __mul__(self, other):  # type: ignore[override]
        # type: (float) -> Translation
        a = float(other)
        return Translation([self.x * a, self.y * a, self.z * a])

    def __rmul__(self, other):  # type: ignore[override]
        # type: (float) -> Translation
        return self * other

    def __imul__(self, other):  # type: ignore[override]
        # type: (float) -> Translation
        self[:] = self * other
        return self

    if six.PY2:

        def __div__(self, other):
            # type: (float) -> Translation
            return self * (1 / float(other))

        def __idiv__(self, other):
            # type: (float) -> Translation
            self[:] = self / other
            return self

    if six.PY3:

        def __truediv__(self, other):
            # type: (float) -> Translation
            return self * (1 / float(other))

        def __itruediv__(self, other):
            # type: (float) -> Translation
            self[:] = self / other
            return self

    def __floordiv__(self, other):
        # type: (Iterable[float]) -> Quaternion
        return CreateQuatRotateDirection(Translation(other), self)

    def __neg__(self):
        # type: () -> Translation
        return self * -1

    @overload  # type: ignore[override]
    def __add__(self, other):
        # type: (Translation) -> Translation
        pass

    @overload
    def __add__(self, other):
        # type: (Union[Quaternion, Pose]) -> Pose
        pass

    @overload
    def __add__(self, other):
        # type: (Iterable[float]) -> Union[Translation, Pose]
        pass

    def __add__(self, other):
        # type: (Iterable[float]) -> Union[Translation, Pose]
        other = list(other)
        if len(other) == 3:
            t = Translation(other)
            return Translation([self.x + t.x, self.y + t.y, self.z + t.z])
        if len(other) == 4:
            q = Quaternion(other)
            return Pose([q.w, q.x, q.y, q.z, self.x, self.y, self.z])
        if len(other) == 7:
            p = Pose(other)
            return self + p.t + p.q
        raise TypeError("Translation does not add with %r" % other)

    def __iadd__(self, other):  # type: ignore[override,misc]
        # type: (Iterable[float]) -> Translation
        other = list(other)
        if len(other) == 3:
            self[:] = self + other
            return self
        raise TypeError("Translation does not add with %r" % other)

    @overload
    def __sub__(self, other):
        # type: (Translation) -> Translation
        pass

    @overload
    def __sub__(self, other):
        # type: (Union[Quaternion, Pose]) -> Pose
        pass

    @overload
    def __sub__(self, other):
        # type: (Iterable[float]) -> Union[Translation, Pose]
        pass

    def __sub__(self, other):
        # type: (Iterable[float]) -> Union[Translation, Pose]
        other = list(other)
        if len(other) == 3:
            return self + (-Translation(other))
        if len(other) == 4:
            return self + (-Quaternion(other))
        if len(other) == 7:
            return self + (-Pose(other))
        raise TypeError("Translation does not subtract with %r" % other)

    def __isub__(self, other):  # type: ignore[misc]
        # type: (Iterable[float]) -> Translation
        other = list(other)
        if len(other) == 3:
            self[:] = self - other
            return self
        raise TypeError("Translation does not subtract with %r" % other)

    def __deepcopy__(self, memodict=None):
        # type: (None) -> Translation
        return Translation(self[:])

    I = None  # type: Translation


Translation.I = Translation([0, 0, 0])

Translate = Translation
T = Translation


class Quaternion(_FixedSizeObservableList):
    def __init__(self, q, normalized=False):
        # type: (Iterable[float], bool) -> None
        if not (isinstance(q, Quaternion) or normalized):
            q = [float(v) for v in q]
            assert len(q) == 4, q
            norm = math.sqrt(sum(v * v for v in q))
            q = [v / norm for v in q]
        super(Quaternion, self).__init__(q)

    @property
    def w(self):
        # type: () -> float
        return self[0]

    @property
    def x(self):
        # type: () -> float
        return self[1]

    @property
    def y(self):
        # type: () -> float
        return self[2]

    @property
    def z(self):
        # type: () -> float
        return self[3]

    @property
    def img(self):
        # type: () -> Translation
        return Translation(self[1:])

    def __neg__(self):
        # type: () -> Quaternion
        return Quaternion([self.w, -self.x, -self.y, -self.z], normalized=True)

    def __mul__(self, other):  # type: ignore[override]
        # type: (Iterable[float]) -> Translation
        if isinstance(other, int):
            raise NotImplementedError()
        return RotateQuatPoint(self, Translation(other))

    @overload  # type: ignore[override]
    def __add__(self, other):
        # type: (Quaternion) -> Quaternion
        pass

    @overload
    def __add__(self, other):
        # type: (Union[Translation, Pose]) -> Pose
        pass

    @overload
    def __add__(self, other):
        # type: (Iterable[float]) -> Union[Quaternion, Pose]
        pass

    def __add__(self, other):
        # type: (Iterable[float]) -> Union[Quaternion, Pose]
        other = list(other)
        if len(other) == 3:
            return RotateQuatPoint(self, Translation(other)) + self
        if len(other) == 4:
            return MultiplyQuat(self, Quaternion(other))
        if len(other) == 7:
            p = Pose(other)
            return RotateQuatPoint(self, p.t) + (self + p.q)
        raise TypeError("Quaternion does not add with %r" % other)

    def __iadd__(self, other):  # type: ignore[override,misc]
        # type: (Iterable[float]) -> Quaternion
        other = list(other)
        if len(other) == 4:
            self[:] = self + other
            return self
        raise TypeError("Quaternion does not add with %r" % other)

    @overload
    def __sub__(self, other):
        # type: (Quaternion) -> Quaternion
        pass

    @overload
    def __sub__(self, other):
        # type: (Union[Translation, Pose]) -> Pose
        pass

    @overload
    def __sub__(self, other):
        # type: (Iterable[float]) -> Union[Quaternion, Pose]
        pass

    def __sub__(self, other):
        # type: (Iterable[float]) -> Union[Quaternion, Pose]
        other = list(other)
        if len(other) == 3:
            return self + (-Translation(other))
        if len(other) == 4:
            return self + (-Quaternion(other))
        if len(other) == 7:
            return self + (-Pose(other))
        raise TypeError("Quaternion does not subtract with %r" % other)

    def __isub__(self, other):  # type: ignore[misc]
        # type: (Iterable[float]) -> Quaternion
        other = list(other)
        if len(other) == 4:
            self[:] = self - other
            return self
        raise TypeError("Quaternion does not subtract with %r" % other)

    @property
    def rotationVector(self):
        # type: () -> Vec3
        d = math.sqrt(sum(v * v for v in self.img))
        theta = 2 * math.atan2(d, self.w)
        if d == 0.0:
            return Vec3.I
        else:
            return Vec3([self.x, self.y, self.z]) / d * theta

    def __deepcopy__(self, memodict=None):
        # type: (None) -> Quaternion
        return Quaternion(self[:])

    I = None  # type: Quaternion


Quaternion.I = Quaternion([1, 0, 0, 0], normalized=True)

Quat = Quaternion
Q = Quaternion


class Pose(_FixedSizeObservableList):
    def __init__(self, p, normalized=False):
        # type: (Iterable[float], bool) -> None
        if not (isinstance(p, Pose) or normalized):
            p = [float(v) for v in p]
            assert len(p) == 7, p
            p[:4] = Quaternion(p[:4])
        super(Pose, self).__init__(p)

    @property
    def t(self):
        # type: () -> Translation
        class Observer(object):
            def onNext(self_, value):
                # type: (list[float]) -> None
                self.t = value

            def onError(self, exc):
                # type: (Exception) -> None
                pass

            def onCompleted(self):
                # type: () -> None
                pass

        t = Translation(self[4:])
        t.Subscribe(Observer())

        return t

    @t.setter
    def t(self, value):
        # type: (Iterable[float]) -> None
        self[4:] = Translation(value)

    @property
    def q(self):
        # type: () -> Quaternion
        class Observer(object):
            def onNext(self_, value):
                # type: (list[float]) -> None
                self.q = value

            def onError(self, exc):
                # type: (Exception) -> None
                pass

            def onCompleted(self):
                # type: () -> None
                pass

        q = Quaternion(self[:4], normalized=True)
        q.Subscribe(Observer())

        return q

    @q.setter
    def q(self, value):
        # type: (Iterable[float]) -> None
        self[:4] = Quaternion(value)

    def __neg__(self):
        # type: () -> Pose
        return Pose(-RotateQuatPoint(-self.q, self.t) - self.q)

    def __add__(self, other):  # type: ignore[override]
        # type: (Iterable[float]) -> Pose
        other = list(other)
        if len(other) == 3:
            return RotateQuatPoint(self.q, Translation(other)) + self
        if len(other) == 4:
            return self + (Translation.I + Quaternion(other))
        if len(other) == 7:
            p = Pose(other)
            return RotateQuatPoint(self.q, p.t) + self.t + (self.q + p.q)
        raise TypeError("Pose does not add with %r" % other)

    def __iadd__(self, other):  # type: ignore[override]
        # type: (Iterable[float]) -> Pose
        other = list(other)
        if len(other) == 7:
            self[:] = self + other
            return self
        raise TypeError("Pose does not add with %r" % other)

    def __sub__(self, other):
        # type: (Iterable[float]) -> Pose
        other = list(other)
        if len(other) == 3:
            return self + (-Translation(other))
        if len(other) == 4:
            return self + (-Quaternion(other))
        if len(other) == 7:
            return self + (-Pose(other))
        raise TypeError("Pose does not subtract with %r" % other)

    def __isub__(self, other):
        # type: (Iterable[float]) -> Pose
        other = list(other)
        if len(other) == 7:
            self[:] = self - other
            return self
        raise TypeError("Pose does not subtract with %r" % other)

    def __deepcopy__(self, memodict):
        # type: (Any) -> Pose
        return Pose(self[:])

    I = None  # type: Pose


Pose.I = Pose([1, 0, 0, 0, 0, 0, 0], normalized=True)
P = Pose
I = Pose.I


def RotateQuatPoint(q, t):
    # type: (Quaternion, Translation) -> Translation
    xx = q.x * q.x
    xy = q.x * q.y
    xz = q.x * q.z
    xw = q.x * q.w
    yy = q.y * q.y
    yz = q.y * q.z
    yw = q.y * q.w
    zz = q.z * q.z
    zw = q.z * q.w
    return Translation(
        [
            2 * ((0.5 - yy - zz) * t.x + (xy - zw) * t.y + (xz + yw) * t.z),
            2 * ((xy + zw) * t.x + (0.5 - xx - zz) * t.y + (yz - xw) * t.z),
            2 * ((xz - yw) * t.x + (yz + xw) * t.y + (0.5 - xx - yy) * t.z),
        ]
    )


def MultiplyQuat(q0, q1):
    # type: (Quaternion, Quaternion) -> Quaternion
    return Quaternion(
        [
            q0.w * q1.w - q0.x * q1.x - q0.y * q1.y - q0.z * q1.z,
            q0.w * q1.x + q0.x * q1.w + q0.y * q1.z - q0.z * q1.y,
            q0.w * q1.y + q0.y * q1.w + q0.z * q1.x - q0.x * q1.z,
            q0.w * q1.z + q0.z * q1.w + q0.x * q1.y - q0.y * q1.x,
        ]
    )


def CreateQuatRotateDirection(t0, t1):
    # type: (Translation, Translation) -> Quaternion
    """
    :param t0:
    :param t1:
    :return: Raise ZeroDivisionError if
      1. t0 is null vector, or
      2. t1 is null vector, or
      3. t0 and t1 are in opposite directions and solutions are not unique
    """
    norm0 = math.sqrt(sum(v * v for v in t0))
    norm1 = math.sqrt(sum(v * v for v in t1))
    return Quaternion(
        [
            t0.x * t1.x + t0.y * t1.y + t0.z * t1.z + norm0 * norm1,
            t0.y * t1.z - t0.z * t1.y,
            t0.z * t1.x - t0.x * t1.z,
            t0.x * t1.y - t0.y * t1.x,
        ]
    )


if serde is not None:

    def _ValidateSpatial(value, type_, ctor, typeArgs, allowTransform):
        # type: (Any, Any, Any, tuple[Any, ...], bool) -> Any
        if isinstance(value, ctor):
            return value
        if allowTransform:
            with contextlib.suppress(AssertionError):
                return ctor(value)
        raise TypeError("%r cannot be coerced into type %s" % (value, type_))

    serde.HASHED_VALIDATORS[Vec3] = _ValidateSpatial
    serde.HASHED_VALIDATORS[Translation] = _ValidateSpatial
    serde.HASHED_VALIDATORS[Quaternion] = _ValidateSpatial
    serde.HASHED_VALIDATORS[Pose] = _ValidateSpatial

__all__ = ["Vec3", "Translation", "Translate", "T", "Quaternion", "Quat", "Q", "Pose", "P", "I"]
