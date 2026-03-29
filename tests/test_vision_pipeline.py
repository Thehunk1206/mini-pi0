import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from mini_pi0.config.io import load_config
from mini_pi0.train.runner import run_train
from mini_pi0.vision.encoders import list_timm_model_options, list_torchvision_model_options
from mini_pi0.vision.precompute import run_precompute_vision


def _create_demo_hdf5(path: Path, t: int = 6) -> None:
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.random.randn(t, 7).astype(np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=np.random.randint(0, 255, size=(t, 64, 64, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eef_pos", data=np.random.randn(t, 3).astype(np.float32))
        obs.create_dataset("robot0_eef_quat", data=np.random.randn(t, 4).astype(np.float32))
        obs.create_dataset("robot0_gripper_qpos", data=np.random.randn(t, 2).astype(np.float32))


class VisionPipelineTests(unittest.TestCase):
    def test_torchvision_model_options_precompute(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h5 = root / "demo.hdf5"
            _create_demo_hdf5(h5)

            for model_name in list_torchvision_model_options():
                out_npz = root / f"tv_{model_name}.npz"
                cfg = load_config(
                    overrides=[
                        f"experiment.runs_root='{str(root / 'runs')}'",
                        f"experiment.name='vision-tv-{model_name}'",
                        "data.format='robomimic_hdf5'",
                        f"data.robomimic_hdf5='{str(h5)}'",
                        "data.robomimic_data_group='data'",
                        "data.n_demos=1",
                        "robot.image_key='agentview_image'",
                        "vision.backend='torchvision'",
                        f"vision.model_name='{model_name}'",
                        "vision.pretrained=false",
                        "vision.batch_size=2",
                        "vision.image_size=64",
                        f"data.precomputed_features_path='{str(out_npz)}'",
                        "train.device='cpu'",
                    ]
                )
                res = run_precompute_vision(cfg)
                self.assertTrue(Path(res["features_path"]).exists())
                with np.load(res["features_path"]) as z:
                    self.assertIn("ep_000000", z.files)
                    self.assertEqual(z["ep_000000"].ndim, 2)
                    self.assertGreater(z["ep_000000"].shape[1], 0)

    def test_timm_model_options_precompute(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h5 = root / "demo.hdf5"
            _create_demo_hdf5(h5)

            # Test the full documented recommended set.
            for model_name in list_timm_model_options(only_recommended=True):
                out_npz = root / f"timm_{model_name.replace('.', '_').replace('/', '_')}.npz"
                cfg = load_config(
                    overrides=[
                        f"experiment.runs_root='{str(root / 'runs')}'",
                        f"experiment.name='vision-timm-{model_name}'",
                        "data.format='robomimic_hdf5'",
                        f"data.robomimic_hdf5='{str(h5)}'",
                        "data.robomimic_data_group='data'",
                        "data.n_demos=1",
                        "robot.image_key='agentview_image'",
                        "vision.backend='timm'",
                        f"vision.model_name='{model_name}'",
                        "vision.pretrained=false",
                        "vision.batch_size=2",
                        "vision.image_size=224",
                        f"data.precomputed_features_path='{str(out_npz)}'",
                        "train.device='cpu'",
                    ]
                )
                res = run_precompute_vision(cfg)
                self.assertTrue(Path(res["features_path"]).exists())
                with np.load(res["features_path"]) as z:
                    self.assertIn("ep_000000", z.files)
                    self.assertEqual(z["ep_000000"].ndim, 2)
                    self.assertGreater(z["ep_000000"].shape[1], 0)

    def test_end_to_end_precomputed_train_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h5 = root / "demo.hdf5"
            _create_demo_hdf5(h5, t=8)
            feats = root / "feats.npz"
            ckpt_dir = root / "ckpt"

            cfg_pre = load_config(
                overrides=[
                    f"experiment.runs_root='{str(root / 'runs')}'",
                    "experiment.name='vision-e2e'",
                    "data.format='robomimic_hdf5'",
                    f"data.robomimic_hdf5='{str(h5)}'",
                    "data.robomimic_data_group='data'",
                    "data.n_demos=1",
                    "robot.image_key='agentview_image'",
                    "vision.backend='torchvision'",
                    "vision.model_name='resnet18'",
                    "vision.pretrained=false",
                    "vision.batch_size=2",
                    "vision.image_size=64",
                    f"data.precomputed_features_path='{str(feats)}'",
                    "train.device='cpu'",
                ]
            )
            run_precompute_vision(cfg_pre)
            self.assertTrue(feats.exists())

            cfg_train = load_config(
                overrides=[
                    f"experiment.runs_root='{str(root / 'runs')}'",
                    "experiment.name='vision-e2e-train'",
                    "data.format='robomimic_hdf5'",
                    f"data.robomimic_hdf5='{str(h5)}'",
                    "data.robomimic_data_group='data'",
                    "data.observation_mode='precomputed'",
                    f"data.precomputed_features_path='{str(feats)}'",
                    "data.precomputed_feature_key='vision_feat'",
                    "data.n_demos=1",
                    "data.chunk_size=4",
                    "model.obs_mode='feature'",
                    "train.epochs=1",
                    "train.batch_size=2",
                    "train.num_workers=0",
                    "train.persistent_workers=false",
                    f"train.ckpt_dir='{str(ckpt_dir)}'",
                    "train.device='cpu'",
                ]
            )
            out = run_train(cfg_train)
            self.assertTrue(Path(out["best_checkpoint"]).exists())


if __name__ == "__main__":
    unittest.main()
