import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cloud_upload import CloudUploadConfig


class CloudUploadConfigTests(unittest.TestCase):
    def test_empty_inherited_values_do_not_mask_dotenv_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / '.env'
            env.write_text('API_BASE_URL=https://example.test\nDEVICE_ID=example\nDEVICE_TOKEN=secret\n')
            with patch('app_config.dotenv_candidate_paths', return_value=(env,)), \
                 patch.dict('os.environ', {'API_BASE_URL': '', 'DEVICE_ID': '', 'DEVICE_TOKEN': ''}):
                config = CloudUploadConfig.from_environment()
                self.assertEqual(config.api_base_url, 'https://example.test')
                self.assertEqual(config.device_id, 'example')
                self.assertEqual(config.device_token, 'secret')

    def test_nonempty_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / '.env'
            env.write_text('API_BASE_URL=https://file.test\nDEVICE_ID=file\nDEVICE_TOKEN=file\n')
            with patch('app_config.dotenv_candidate_paths', return_value=(env,)), \
                 patch.dict('os.environ', {'API_BASE_URL': 'https://env.test', 'DEVICE_ID': 'env', 'DEVICE_TOKEN': 'env'}):
                config = CloudUploadConfig.from_environment()
                self.assertEqual(config.api_base_url, 'https://env.test')
                self.assertEqual(config.device_id, 'env')
                self.assertEqual(config.device_token, 'env')
