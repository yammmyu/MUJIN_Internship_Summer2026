import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from openravepy import CollisionOptions, CollisionReport, Environment, GeometryType, IkFilterOptions, IkParameterization, IkParameterizationType, KinBody, Manipulator, Robot
from openravepy.databases import inversekinematics

from .spatial import Pose as P
from .spatial import Translation as T

log = logging.getLogger(__name__)


def FilterCollidingLinkPairs(
    robot,  # type: Robot
    target,  # type: Optional[KinBody]
    linkPairs,  # type: List[Tuple[KinBody.Link, KinBody.Link]]
):
    # type: (...) -> List[Tuple[KinBody.Link, KinBody.Link]]
    rootLink = next(l for l in robot.GetLinks() if not l.GetParentLinks())

    def IsRobotBaseCollision(link1, link2):
        # type: (KinBody.Link, KinBody.Link) -> bool
        return (
            link1 == rootLink
                and link2.GetParent() != robot
                and link2.GetParent() not in robot.GetGrabbed()  # fmt: skip
        )

    def IsGrabbedIgnoredCollision(link):
        # type:(KinBody.Link) -> bool
        return bool(link.GetParent() == target)

    return [
        (l1, l2) if l1.GetParent() == robot else (l2, l1)
        for l1, l2 in linkPairs
        if not (
            IsRobotBaseCollision(l1, l2)
                or IsRobotBaseCollision(l2, l1)
                or IsGrabbedIgnoredCollision(l1)
                or IsGrabbedIgnoredCollision(l2)  # fmt: skip
        )
    ]


class IkFailureJointLimits(object):
    def __repr__(self):  # type: () -> str
        return "%s()" % self.__class__.__name__


class IkFailureCustomFilters(object):
    def __repr__(self):  # type: () -> str
        return "%s()" % self.__class__.__name__


class IkFailureCollision(object):
    def __init__(self, linkPairs):
        # type: (List[Tuple[KinBody.Link, KinBody.Link]]) -> None
        self.linkPairs = linkPairs

    def __repr__(self):
        # type: (...) -> str
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(
                "(%s:%s)x(%s:%s)"
                % (
                    link1.GetParent().GetName(),
                    link1.GetName(),
                    link2.GetParent().GetName(),
                    link2.GetName(),
                )
                for link1, link2 in self.linkPairs
            ),
        )


class IkFailureSelfCollision(IkFailureCollision):
    pass


class AltitudeFailure(object):
    def __init__(self, degree):
        # type: (float) -> None
        self.degree = degree

    def __repr__(self):
        # type: (...) -> str
        return "%s(%.2f)" % (self.__class__.__name__, self.degree)


# if typing.TYPE_CHECKING:
IkFailure = Union[IkFailureSelfCollision, IkFailureJointLimits, IkFailureCustomFilters, IkFailureCollision, AltitudeFailure]  # fmt: skip


def _GetSolutionHash(solution):
    # type: (np.ndarray) -> str
    return ",".join("%.4f" % np.mod(v, np.pi * 2) for v in solution)


def FindIkSolutions(
    manip,  # type: Manipulator
    ikParam,  # type: Union[List[float], IkParameterization]
    target=None,  # type: Optional[KinBody]
):  # type: (...) -> List[Tuple[List[float], List[IkFailure]]]
    robot = manip.GetRobot()
    env = robot.GetEnv()

    if isinstance(ikParam, list):
        if len(ikParam) == 7:
            ikParam = IkParameterization(P(ikParam), IkParameterizationType.Transform6D)
        elif len(ikParam) == 3:
            ikParam = IkParameterization(T(ikParam), IkParameterizationType.Translation3D)
        else:
            raise NotImplementedError
    ikmodel = inversekinematics.InverseKinematicsModel(manip=manip, iktype=ikParam.GetType())  # fmt: skip
    if not ikmodel.load():
        log.error("Failed to load ikmodel. Generating...")
        ikmodel.autogenerate()

    with robot.CreateRobotStateSaver():
        preshape = ikParam.GetCustomValues("preshape")
        gripperDofIndices = manip.GetGripperIndices()
        if preshape is not None and len(preshape) == len(gripperDofIndices):
            robot.SetDOFValues(np.array(preshape) * 1e-3, gripperDofIndices)

        reachableSolutions = {
            _GetSolutionHash(solution): (solution.tolist(), [])
            for solution in manip.FindIKSolutions(ikParam, 0b1111110)  # fmt: skip
        }  # type: Dict[str, Tuple[List[float], List[IkFailure]]]

        cr = CollisionReport()
        cc = robot.GetEnv().GetCollisionChecker()
        cc.SetCollisionOptions(CollisionOptions.AllLinkCollisions)

        for bit, constraint in [
            (IkFilterOptions.IgnoreJointLimits, IkFailureJointLimits()),
            (
                IkFilterOptions.CheckEnvCollisions
                    | IkFilterOptions.IgnoreSelfCollisions
                    | IkFilterOptions.IgnoreEndEffectorCollisions
                    | IkFilterOptions.IgnoreEndEffectorEnvCollisions
                    | IkFilterOptions.IgnoreEndEffectorSelfCollisions,
                    IkFailureCollision([]),  # fmt: skip
            ),
            (IkFilterOptions.IgnoreCustomFilters, IkFailureCustomFilters()),
        ]:
            availableSolutions = {
                _GetSolutionHash(solution): solution.tolist()
                for solution in manip.FindIKSolutions(ikParam, bit ^ 0b1111110)  # fmt: skip
            }
            for solutionHash, (solution, constraints) in reachableSolutions.items():
                if solutionHash in availableSolutions:
                    if isinstance(constraint, IkFailureJointLimits):
                        solution[:] = availableSolutions[solutionHash]
                elif isinstance(constraint, IkFailureCollision):
                    if any(isinstance(constraint, IkFailureJointLimits) for constraint in constraints):  # fmt: skip
                        continue

                    with robot.CreateRobotStateSaver():
                        # Need to separate inter-collision and self-collision
                        robot.SetDOFValues(solution, range(len(solution)))

                        if robot.CheckSelfCollision(cr):
                            vLinkColliding = [(env.GetKinBody(info.ExtractFirstBodyLinkGeomNames()[0]).GetLink(info.ExtractFirstBodyLinkGeomNames()[1]), env.GetKinBody(info.ExtractSecondBodyLinkGeomNames()[0]).GetLink(info.ExtractSecondBodyLinkGeomNames()[1])) for info in cr.collisionInfos]
                            constraints.append(IkFailureSelfCollision(vLinkColliding))

                        cc.CheckCollision(robot, cr)
                        vLinkColliding = [(env.GetKinBody(info.ExtractFirstBodyLinkGeomNames()[0]).GetLink(info.ExtractFirstBodyLinkGeomNames()[1]), env.GetKinBody(info.ExtractSecondBodyLinkGeomNames()[0]).GetLink(info.ExtractSecondBodyLinkGeomNames()[1])) for info in cr.collisionInfos]
                        linkPairs = FilterCollidingLinkPairs(robot, target, vLinkColliding)  # fmt: skip
                        if linkPairs:
                            constraints.append(IkFailureCollision(linkPairs))
                else:
                    constraints.append(constraint)

    return sorted(list(reachableSolutions.values()))
