import unittest
from webapp import sanitize_diagnostic

class DiagnosticsPrivacyTests(unittest.TestCase):
    def test_removes_secrets_and_credentials(self):
        source = {'password':'secret', 'token':'token', 'rtsp_url':'rtsp://user:pass@host',
                  'status':'failed rtsp://user:pass@host/stream password=bad',
                  'nested':[{'read_url':'https://host/?sig=secret', 'camera':'front'}]}
        result = sanitize_diagnostic(source)
        self.assertEqual(result, {'status':'failed [redacted stream] password=[redacted]',
                                  'nested':[{'camera':'front'}]})
