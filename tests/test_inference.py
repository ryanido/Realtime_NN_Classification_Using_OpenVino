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

class TestGetInferenceInput(unittest.TestCase):
    def test_get_inference_input(self):
        # Create a sample image with shape (416, 416, 3)
        image = np.random.rand(416, 416, 3)

        # Call the function with input_size=224
        input_data = get_inference_input(image, 224)

        # Check that the keys in the output dictionary are correct
        self.assertCountEqual(list(input_data.keys()), ['input_1', 'image_shape'])

        # Check that the input data has the correct shape and data type
        self.assertEqual(input_data['input_1'].shape, (1, 3, 224, 224))
        self.assertEqual(input_data['input_1'].dtype, np.float32)

        # Check that the image shape has been correctly stored
        self.assertEqual(input_data['image_shape'].shape, (1, 2))
        self.assertEqual(input_data['image_shape'].dtype, np.float32)
        self.assertEqual(list(input_data['image_shape'][0]), [416, 416])

if __name__ == '__main__':
    unittest.main()
