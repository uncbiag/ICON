import unittest
import subprocess
from icon_registration import test_utils

class TestConsoleScripts(unittest.TestCase):
    def test_register_pair_script(self):
        import footsteps
        test_utils.download_test_data()
        subprocess.run(
          [
            "icon_register_pair",
            "--fixed_image", 
            test_utils.TEST_DATA_DIR / "brain_test_data" / "2_T1w_acpc_dc_restore_brain.nii.gz",
            "--moving_image", 
            test_utils.TEST_DATA_DIR / "brain_test_data" / "8_T1w_acpc_dc_restore_brain.nii.gz",
            "--model",
            "icon_registration.pretrained_models.brain_registration_model"
            "--warped_image_out", 
            footsteps.output_dir + "warped.nii.gz"
          ]
        )
        import itk
        itk.imread(footsteps.output_dir + "warped.nii.gz")
