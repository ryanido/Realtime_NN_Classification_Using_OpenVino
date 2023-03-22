import unittest
import numpy as np
import sys
sys.path.append("src/")
from get_inference_input import * 

#Tests for inference helper functions
class TestResizeImageLetterbox(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 50, 3), dtype=np.uint8)
        self.size = (200, 200)

    def test_resize_image_letterbox(self):
        resized_image = resize_image_letterbox(self.image, self.size)
        self.assertEqual(resized_image.shape[0], self.size[1])
        self.assertEqual(resized_image.shape[1], self.size[0])  
if __name__ == '__main__':
    unittest.main()
