"""Optional phone-control modes layered onto LeRobot's processor pipeline."""

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction
from lerobot.processor import RobotActionProcessorStep


ROTATION_TARGET_KEYS = ("target_wx", "target_wy", "target_wz")


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
