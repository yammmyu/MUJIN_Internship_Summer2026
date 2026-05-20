"""
Capture HMD and left/right controller 6DoF poses, trigger state, button
clicks, and joystick (thumbstick) state via OpenXR.

Mirrors the relevant pieces of ``xr_examples/hello_xr/openxr_program.py``
(pose action, grab/trigger action, hand subaction paths, action spaces,
interaction-profile bindings, per-frame ``sync_actions`` + ``locate_space``)
and additionally samples the headset by locating its built-in ``VIEW``
reference space relative to a ``LOCAL`` reference space.

Each :meth:`PoseInput.sample` returns a :class:`PoseSample` carrying:

* head / left / right poses as :class:`xr.Posef` (identity if untracked) plus
  tracked-flag bools, for in-scene rendering
* ``left_grab`` / ``right_grab`` bools (analog trigger thresholded at
  :attr:`PoseInput.grab_threshold`)
* Oculus-Touch face buttons: ``left_x``, ``left_y``, ``right_a``, ``right_b``
  (digital click bools; ``False`` on controllers that don't expose them)
* Thumbstick state per hand: ``left_stick``, ``right_stick`` as
  ``(x, y)`` tuples in ``[-1, 1]^2`` and ``left_stick_click`` /
  ``right_stick_click`` bools
* a ``timestamp`` (monotonic seconds)

Call :meth:`PoseSample.to_joint_state` to pack the snapshot into the wire
:class:`JointState` (21 floats in ``positions``, 8 bools in ``triggers``,
4 floats in ``axes``)::

    positions = [head_p xyz, head_q xyzw,
                 left_p xyz, left_q xyzw,
                 right_p xyz, right_q xyzw]
    triggers  = [left_grab, right_grab,
                 left_x, left_y, left_stick_click,
                 right_a, right_b, right_stick_click]
    axes      = [left_stick_x, left_stick_y,
                 right_stick_x, right_stick_y]

Coordinates are in metres relative to the ``LOCAL`` reference space owned by
this module (origin at headset position when the session started, Y up).
Quaternion components follow OpenXR's ``XrPosef.orientation`` order:
``(x, y, z, w)`` with ``w`` last.
"""

import logging
from ctypes import pointer
from dataclasses import dataclass
from typing import Optional, Tuple

import xr

from pico_vr.pico_vr_common.protocol import JointState, monotonic_timestamp

logger = logging.getLogger("xr_examples.pico_vr_client.pose_input")

JOINT_NAMES: Tuple[str, ...] = (
    "head_px", "head_py", "head_pz",
    "head_qx", "head_qy", "head_qz", "head_qw",
    "left_px", "left_py", "left_pz",
    "left_qx", "left_qy", "left_qz", "left_qw",
    "right_px", "right_py", "right_pz",
    "right_qx", "right_qy", "right_qz", "right_qw",
)

# For each interaction profile we list the input component that drives the
# grab action. KHR-simple only exposes /select/click (digital), while modern
# controllers expose an analog /trigger/value that the runtime promotes to
# our FLOAT_INPUT action. The pose action is always bound to /grip/pose.
_PROFILE_GRAB_COMPONENT = {
    "/interaction_profiles/khr/simple_controller": "input/select/click",
    "/interaction_profiles/htc/vive_controller": "input/trigger/value",
    "/interaction_profiles/oculus/touch_controller": "input/trigger/value",
    "/interaction_profiles/valve/index_controller": "input/trigger/value",
}

# Per-hand face/joystick bindings, suggested only on profiles that actually
# expose them. Profiles not listed here just leave the corresponding actions
# inactive at runtime and PoseInput reports False / 0.0 for them.
#   left: X / Y / thumbstick (V2f + click)
#   right: A / B / thumbstick (V2f + click)
_PROFILE_BUTTONS = {
    # Oculus Touch is the canonical mapping for these names.
    "/interaction_profiles/oculus/touch_controller": {
        "left_x": "/user/hand/left/input/x/click",
        "left_y": "/user/hand/left/input/y/click",
        "left_stick": "/user/hand/left/input/thumbstick",
        "left_stick_click": "/user/hand/left/input/thumbstick/click",
        "right_a": "/user/hand/right/input/a/click",
        "right_b": "/user/hand/right/input/b/click",
        "right_stick": "/user/hand/right/input/thumbstick",
        "right_stick_click": "/user/hand/right/input/thumbstick/click",
    },
    # Valve Index has A/B on both hands and thumbsticks on both, but no X/Y.
    # We map A/B → the per-side button names (left A/B → left_x/left_y so the
    # operator still has two configurable face buttons per hand).
    "/interaction_profiles/valve/index_controller": {
        "left_x": "/user/hand/left/input/a/click",
        "left_y": "/user/hand/left/input/b/click",
        "left_stick": "/user/hand/left/input/thumbstick",
        "left_stick_click": "/user/hand/left/input/thumbstick/click",
        "right_a": "/user/hand/right/input/a/click",
        "right_b": "/user/hand/right/input/b/click",
        "right_stick": "/user/hand/right/input/thumbstick",
        "right_stick_click": "/user/hand/right/input/thumbstick/click",
    },
}


def _identity_posef() -> xr.Posef:
    """Return a fresh identity xr.Posef (orientation.w defaults to 1)."""
    return xr.Posef()


def _copy_posef(src: xr.Posef) -> xr.Posef:
    """Copy an xr.Posef into a freshly allocated struct, decoupling lifetimes."""
    dst = xr.Posef()
    dst.position.x = src.position.x
    dst.position.y = src.position.y
    dst.position.z = src.position.z
    dst.orientation.x = src.orientation.x
    dst.orientation.y = src.orientation.y
    dst.orientation.z = src.orientation.z
    dst.orientation.w = src.orientation.w
    return dst


def _pose_to_floats(p: xr.Posef) -> Tuple[float, float, float, float, float, float, float]:
    return (
        p.position.x, p.position.y, p.position.z,
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
    )


@dataclass
class PoseSample:
    """One frame's worth of XR input — poses, tracking flags, buttons, sticks."""

    timestamp: float
    head: xr.Posef
    head_tracked: bool
    left: xr.Posef
    left_tracked: bool
    right: xr.Posef
    right_tracked: bool
    left_grab: bool
    right_grab: bool
    left_x: bool = False
    left_y: bool = False
    right_a: bool = False
    right_b: bool = False
    left_stick: Tuple[float, float] = (0.0, 0.0)
    right_stick: Tuple[float, float] = (0.0, 0.0)
    left_stick_click: bool = False
    right_stick_click: bool = False

    def to_joint_state(self) -> JointState:
        return JointState(
            timestamp=self.timestamp,
            names=list(JOINT_NAMES),
            positions=list(_pose_to_floats(self.head))
            + list(_pose_to_floats(self.left))
            + list(_pose_to_floats(self.right)),
            triggers=[
                self.left_grab, self.right_grab,
                self.left_x, self.left_y, self.left_stick_click,
                self.right_a, self.right_b, self.right_stick_click,
            ],
            axes=[
                self.left_stick[0], self.left_stick[1],
                self.right_stick[0], self.right_stick[1],
            ],
        )


class PoseInput:
    """Owns the pose action, grab action, head/hand spaces, and a LOCAL anchor."""

    def __init__(self, instance: xr.Instance, session: xr.Session,
                 action_set: xr.ActionSet, grab_threshold: float = 0.5) -> None:
        self.instance = instance
        self.session = session
        self.action_set = action_set
        self.grab_threshold = grab_threshold
        self.pose_action: Optional[xr.Action] = None
        self.grab_action: Optional[xr.Action] = None
        # Per-hand digital face buttons. Bound only on profiles that expose
        # them; otherwise stay inactive and read as False.
        self.left_x_action: Optional[xr.Action] = None
        self.left_y_action: Optional[xr.Action] = None
        self.right_a_action: Optional[xr.Action] = None
        self.right_b_action: Optional[xr.Action] = None
        # Thumbstick V2f + click, one action of each shared across both hands
        # (subaction path picks the side at sample time).
        self.stick_action: Optional[xr.Action] = None
        self.stick_click_action: Optional[xr.Action] = None
        self.left_subaction_path = xr.NULL_PATH
        self.right_subaction_path = xr.NULL_PATH
        self.left_space: Optional[xr.Space] = None
        self.right_space: Optional[xr.Space] = None
        self.head_space: Optional[xr.Space] = None  # VIEW reference space
        self.local_space: Optional[xr.Space] = None  # LOCAL reference space
        self._first_head_logged = False
        self._first_left_logged = False
        self._first_right_logged = False
        self._first_left_grab_logged = False
        self._first_right_grab_logged = False

    def initialize(self) -> None:
        self.left_subaction_path = xr.string_to_path(self.instance, "/user/hand/left")
        self.right_subaction_path = xr.string_to_path(self.instance, "/user/hand/right")
        subaction_paths = (xr.Path * 2)(
            self.left_subaction_path, self.right_subaction_path,
        )
        self.pose_action = xr.create_action(
            action_set=self.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="hand_pose",
                localized_action_name="Hand Pose",
                count_subaction_paths=2,
                subaction_paths=subaction_paths,
            ),
        )
        self.grab_action = xr.create_action(
            action_set=self.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.FLOAT_INPUT,
                action_name="grab_object",
                localized_action_name="Grab Object",
                count_subaction_paths=2,
                subaction_paths=subaction_paths,
            ),
        )
        # Face buttons: per-hand boolean actions. We use separate actions
        # (rather than one with two bindings) so each click is independently
        # observable from the action state.
        self.left_x_action = self._make_bool_action(
            "left_x", "Left X Button", (self.left_subaction_path,),
        )
        self.left_y_action = self._make_bool_action(
            "left_y", "Left Y Button", (self.left_subaction_path,),
        )
        self.right_a_action = self._make_bool_action(
            "right_a", "Right A Button", (self.right_subaction_path,),
        )
        self.right_b_action = self._make_bool_action(
            "right_b", "Right B Button", (self.right_subaction_path,),
        )
        # Thumbstick: shared V2f and click actions, indexed by subaction path.
        self.stick_action = xr.create_action(
            action_set=self.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.VECTOR2F_INPUT,
                action_name="thumbstick",
                localized_action_name="Thumbstick",
                count_subaction_paths=2,
                subaction_paths=subaction_paths,
            ),
        )
        self.stick_click_action = xr.create_action(
            action_set=self.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.BOOLEAN_INPUT,
                action_name="thumbstick_click",
                localized_action_name="Thumbstick Click",
                count_subaction_paths=2,
                subaction_paths=subaction_paths,
            ),
        )
        self._suggest_bindings()

        self.left_space = xr.create_action_space(
            session=self.session,
            create_info=xr.ActionSpaceCreateInfo(
                action=self.pose_action,
                subaction_path=self.left_subaction_path,
            ),
        )
        self.right_space = xr.create_action_space(
            session=self.session,
            create_info=xr.ActionSpaceCreateInfo(
                action=self.pose_action,
                subaction_path=self.right_subaction_path,
            ),
        )
        # Head pose = pose of the VIEW reference space relative to LOCAL.
        # Owning our own copies means this class is independent of whatever
        # reference space the caller configured ContextObject with.
        self.head_space = xr.create_reference_space(
            session=self.session,
            create_info=xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.VIEW,
            ),
        )
        self.local_space = xr.create_reference_space(
            session=self.session,
            create_info=xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.LOCAL,
            ),
        )
        logger.info("Pose input initialised (head + left + right, grab action)")

    def _make_bool_action(self, name: str, localized: str,
                           subaction_paths: Tuple[xr.Path, ...]) -> xr.Action:
        sub = (xr.Path * len(subaction_paths))(*subaction_paths)
        return xr.create_action(
            action_set=self.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.BOOLEAN_INPUT,
                action_name=name,
                localized_action_name=localized,
                count_subaction_paths=len(subaction_paths),
                subaction_paths=sub,
            ),
        )

    def _suggest_bindings(self) -> None:
        left_grip = xr.string_to_path(self.instance, "/user/hand/left/input/grip/pose")
        right_grip = xr.string_to_path(self.instance, "/user/hand/right/input/grip/pose")
        for profile, grab_component in _PROFILE_GRAB_COMPONENT.items():
            left_grab = xr.string_to_path(self.instance, f"/user/hand/left/{grab_component}")
            right_grab = xr.string_to_path(self.instance, f"/user/hand/right/{grab_component}")
            pairs = [
                (self.pose_action, left_grip),
                (self.pose_action, right_grip),
                (self.grab_action, left_grab),
                (self.grab_action, right_grab),
            ]
            # Profile-specific face buttons and thumbsticks (Oculus Touch / Index).
            buttons = _PROFILE_BUTTONS.get(profile)
            if buttons is not None:
                pairs.extend([
                    (self.left_x_action, xr.string_to_path(self.instance, buttons["left_x"])),
                    (self.left_y_action, xr.string_to_path(self.instance, buttons["left_y"])),
                    (self.right_a_action, xr.string_to_path(self.instance, buttons["right_a"])),
                    (self.right_b_action, xr.string_to_path(self.instance, buttons["right_b"])),
                    (self.stick_action, xr.string_to_path(self.instance, buttons["left_stick"])),
                    (self.stick_action, xr.string_to_path(self.instance, buttons["right_stick"])),
                    (self.stick_click_action,
                     xr.string_to_path(self.instance, buttons["left_stick_click"])),
                    (self.stick_click_action,
                     xr.string_to_path(self.instance, buttons["right_stick_click"])),
                ])
            bindings = (xr.ActionSuggestedBinding * len(pairs))(
                *(xr.ActionSuggestedBinding(action, path) for action, path in pairs)
            )
            try:
                xr.suggest_interaction_profile_bindings(
                    instance=self.instance,
                    suggested_bindings=xr.InteractionProfileSuggestedBinding(
                        interaction_profile=xr.string_to_path(self.instance, profile),
                        count_suggested_bindings=len(pairs),
                        suggested_bindings=bindings,
                    ),
                )
            except xr.XrException as exc:
                logger.debug(f"Could not suggest bindings for {profile}: {exc}")

    def sync(self) -> bool:
        """Refresh action state. Returns False on SessionNotFocused (safe to call)."""
        active = xr.ActiveActionSet(self.action_set, xr.NULL_PATH)
        try:
            xr.sync_actions(
                self.session,
                xr.ActionsSyncInfo(
                    count_active_action_sets=1,
                    active_action_sets=pointer(active),
                ),
            )
            return True
        except xr.exception.SessionNotFocused:
            return False
        except xr.XrException as exc:
            logger.debug(f"sync_actions failed: {exc}")
            return False

    def sample(self, predicted_display_time: xr.Time) -> PoseSample:
        """Locate head + both controllers, read all action state, return a PoseSample."""
        head, head_tracked = self._locate_space(self.head_space, predicted_display_time)
        left, left_tracked = self._locate_hand(
            self.left_space, self.left_subaction_path, predicted_display_time,
        )
        right, right_tracked = self._locate_hand(
            self.right_space, self.right_subaction_path, predicted_display_time,
        )
        left_grab = self._sample_grab(self.left_subaction_path)
        right_grab = self._sample_grab(self.right_subaction_path)
        left_x = self._sample_bool(self.left_x_action, self.left_subaction_path)
        left_y = self._sample_bool(self.left_y_action, self.left_subaction_path)
        right_a = self._sample_bool(self.right_a_action, self.right_subaction_path)
        right_b = self._sample_bool(self.right_b_action, self.right_subaction_path)
        left_stick = self._sample_vec2(self.stick_action, self.left_subaction_path)
        right_stick = self._sample_vec2(self.stick_action, self.right_subaction_path)
        left_stick_click = self._sample_bool(
            self.stick_click_action, self.left_subaction_path,
        )
        right_stick_click = self._sample_bool(
            self.stick_click_action, self.right_subaction_path,
        )
        self._maybe_log_first_seen(head, head_tracked, left, left_tracked,
                                    right, right_tracked, left_grab, right_grab)
        return PoseSample(
            timestamp=monotonic_timestamp(),
            head=head, head_tracked=head_tracked,
            left=left, left_tracked=left_tracked,
            right=right, right_tracked=right_tracked,
            left_grab=left_grab, right_grab=right_grab,
            left_x=left_x, left_y=left_y,
            right_a=right_a, right_b=right_b,
            left_stick=left_stick, right_stick=right_stick,
            left_stick_click=left_stick_click,
            right_stick_click=right_stick_click,
        )

    def _maybe_log_first_seen(self, head, head_tracked, left, left_tracked,
                               right, right_tracked, left_grab, right_grab) -> None:
        if head_tracked and not self._first_head_logged:
            self._first_head_logged = True
            logger.info(
                f"Head pose acquired: pos=({head.position.x:+.3f},"
                f"{head.position.y:+.3f},{head.position.z:+.3f})"
            )
        if left_tracked and not self._first_left_logged:
            self._first_left_logged = True
            logger.info(
                f"Left controller pose acquired: pos=({left.position.x:+.3f},"
                f"{left.position.y:+.3f},{left.position.z:+.3f})"
            )
        if right_tracked and not self._first_right_logged:
            self._first_right_logged = True
            logger.info(
                f"Right controller pose acquired: pos=({right.position.x:+.3f},"
                f"{right.position.y:+.3f},{right.position.z:+.3f})"
            )
        if left_grab and not self._first_left_grab_logged:
            self._first_left_grab_logged = True
            logger.info("Left trigger pressed (first time)")
        if right_grab and not self._first_right_grab_logged:
            self._first_right_grab_logged = True
            logger.info("Right trigger pressed (first time)")

    def _sample_grab(self, subaction_path: xr.Path) -> bool:
        if self.grab_action is None:
            return False
        try:
            state = xr.get_action_state_float(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=self.grab_action,
                    subaction_path=subaction_path,
                ),
            )
        except xr.XrException as exc:
            logger.debug(f"get_action_state_float (grab) failed: {exc}")
            return False
        if not state.is_active:
            return False
        return float(state.current_state) >= self.grab_threshold

    def _sample_bool(self, action: Optional[xr.Action],
                     subaction_path: xr.Path) -> bool:
        if action is None:
            return False
        try:
            state = xr.get_action_state_boolean(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=action,
                    subaction_path=subaction_path,
                ),
            )
        except xr.XrException as exc:
            logger.debug(f"get_action_state_boolean failed: {exc}")
            return False
        if not state.is_active:
            return False
        return bool(state.current_state)

    def _sample_vec2(self, action: Optional[xr.Action],
                     subaction_path: xr.Path) -> Tuple[float, float]:
        if action is None:
            return (0.0, 0.0)
        try:
            state = xr.get_action_state_vector2f(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=action,
                    subaction_path=subaction_path,
                ),
            )
        except xr.XrException as exc:
            logger.debug(f"get_action_state_vector2f failed: {exc}")
            return (0.0, 0.0)
        if not state.is_active:
            return (0.0, 0.0)
        return (float(state.current_state.x), float(state.current_state.y))

    def _locate_hand(self, hand_space: Optional[xr.Space], subaction_path: xr.Path,
                      predicted_display_time: xr.Time) -> Tuple[xr.Posef, bool]:
        if hand_space is None or self.pose_action is None or self.local_space is None:
            return _identity_posef(), False
        try:
            state = xr.get_action_state_pose(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=self.pose_action,
                    subaction_path=subaction_path,
                ),
            )
            if not state.is_active:
                return _identity_posef(), False
        except xr.XrException as exc:
            logger.debug(f"get_action_state_pose failed: {exc}")
            return _identity_posef(), False
        return self._locate_space(hand_space, predicted_display_time)

    def _locate_space(self, space: Optional[xr.Space],
                       predicted_display_time: xr.Time) -> Tuple[xr.Posef, bool]:
        if space is None or self.local_space is None:
            return _identity_posef(), False
        try:
            location = xr.locate_space(
                space=space,
                base_space=self.local_space,
                time=predicted_display_time,
            )
        except xr.XrException as exc:
            logger.debug(f"locate_space failed: {exc}")
            return _identity_posef(), False
        flags = location.location_flags
        if (flags & xr.SPACE_LOCATION_POSITION_VALID_BIT == 0
                or flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT == 0):
            return _identity_posef(), False
        return _copy_posef(location.pose), True

    def destroy(self) -> None:
        for name in ("left_space", "right_space", "head_space", "local_space"):
            space = getattr(self, name)
            if space is not None:
                try:
                    xr.destroy_space(space)
                except xr.XrException:
                    pass
                setattr(self, name, None)
        # Actions are owned by the action_set, which ContextObject destroys.
        self.pose_action = None
        self.grab_action = None
        self.left_x_action = None
        self.left_y_action = None
        self.right_a_action = None
        self.right_b_action = None
        self.stick_action = None
        self.stick_click_action = None
