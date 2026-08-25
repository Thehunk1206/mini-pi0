"""Optional phone-control modes layered onto LeRobot's processor pipeline."""

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction
from lerobot.processor import RobotActionProcessorStep


ROTATION_TARGET_KEYS = ("target_wx", "target_wy", "target_wz")
TRANSLATION_AXES = ("x", "y", "z")


class RemapPhoneTranslation(RobotActionProcessorStep):
    """Map LeRobot's phone translation outputs into the robot base frame."""

    def __init__(
        self,
        axis_map: dict[str, str],
        axis_signs: dict[str, float],
    ) -> None:
        if set(axis_map) != set(TRANSLATION_AXES):
            raise ValueError("Translation axis map must define x, y, and z outputs")
        if set(axis_map.values()) != set(TRANSLATION_AXES):
            raise ValueError("Translation axis map must use x, y, and z exactly once")
        if set(axis_signs) != set(TRANSLATION_AXES):
            raise ValueError("Translation axis signs must define x, y, and z")
        if any(float(sign) not in {-1.0, 1.0} for sign in axis_signs.values()):
            raise ValueError("Translation axis signs must be either -1 or 1")
        self.axis_map = dict(axis_map)
        self.axis_signs = {axis: float(sign) for axis, sign in axis_signs.items()}

    def action(self, action: RobotAction) -> RobotAction:
        source = {
            axis: float(action[f"target_{axis}"]) for axis in TRANSLATION_AXES
        }
        for output_axis in TRANSLATION_AXES:
            input_axis = self.axis_map[output_axis]
            action[f"target_{output_axis}"] = (
                self.axis_signs[output_axis] * source[input_axis]
            )
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


class DisablePhoneOrientation(RobotActionProcessorStep):
    """Keep the latched robot orientation while allowing XYZ phone motion."""

    def action(self, action: RobotAction) -> RobotAction:
        missing = [key for key in ROTATION_TARGET_KEYS if key not in action]
        if missing:
            raise ValueError(
                "Phone action is missing orientation component(s): "
                + ", ".join(missing)
            )
        for key in ROTATION_TARGET_KEYS:
            action[key] = 0.0
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
