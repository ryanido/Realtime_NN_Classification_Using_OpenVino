import unittest
# from inference import Inference


class TestInference(unittest.TestCase):
    def test_infer(self, mock_IECore):
        self.assertEqual("Hello World", "Hello World")

unittest.main()