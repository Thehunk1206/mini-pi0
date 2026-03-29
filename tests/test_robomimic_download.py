import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_pi0.dataset.robomimic_download import download_robomimic_dataset


class RoboMimicDownloadTests(unittest.TestCase):
    def test_download_metadata_and_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def _fake_download(url, destination, overwrite):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"test")

            with patch("mini_pi0.dataset.robomimic_download._download_file", side_effect=_fake_download):
                out = download_robomimic_dataset(
                    task="lift",
                    dataset_type="ph",
                    hdf5_type="low_dim",
                    download_dir=str(root),
                    version="v1.5",
                    overwrite=False,
                )

            self.assertEqual(out["task"], "lift")
            self.assertEqual(out["dataset_type"], "ph")
            self.assertEqual(out["hdf5_type"], "low_dim")
            self.assertTrue(str(out["path"]).endswith("lift/ph/low_dim_v15.hdf5"))
            self.assertGreater(out["size_bytes"], 0)

    def test_invalid_combo_raises(self):
        with self.assertRaises(ValueError):
            download_robomimic_dataset(
                task="tool_hang",
                dataset_type="mg",
                hdf5_type="low_dim",
                download_dir="data/robomimic",
                version="v1.5",
            )


if __name__ == "__main__":
    unittest.main()
