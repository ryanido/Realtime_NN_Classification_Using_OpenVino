import unittest
import numpy as np
import sys
sys.path.append("src/")
import mobilenet_v2_live_get_inference_input as mn
import mobilenet_v2_live_put_inference_output as mn2
import yolo_v3_live_get_inference_input as yolo
import cv2



#Tests for inference helper functions
class TestResizeImageLetterbox(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 50, 3), dtype=np.uint8)
        self.size = (200, 200)

    def test_resize_image_letterbox(self):
        resized_image = yolo.resize_image_letterbox(self.image, self.size)
        self.assertEqual(resized_image.shape[0], self.size[1])
        self.assertEqual(resized_image.shape[1], self.size[0])

if __name__ == '__main__':
    unittest.main()