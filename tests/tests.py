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

class TestMNGetInferenceInput(unittest.TestCase):
    def test_mn_get_inference_input(self):
        # Create a sample image
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[10:20, 10:20, :] = 255

        # Call the function with some input parameters
        input_size = 32
        input_name = 'data'
        input_data = mn.get_inference_input(image, input_size, input_name)

        # Check that the input data has the expected shape and type
        self.assertEqual(input_data[input_name].shape, (1, 3, input_size, input_size))
        self.assertEqual(input_data[input_name].dtype, np.float32)

        # Check that the input data has been correctly preprocessed
        resized_image = cv2.resize(image, (input_size, input_size))
        expected_data = np.transpose(resized_image, (2, 0, 1)).reshape(1, 3, input_size, input_size).astype(np.float32)
        np.testing.assert_allclose(input_data[input_name], expected_data)
        
class TestGetMostProbableBounds(unittest.TestCase):
    def test_get_most_probable_bounds(self):
        # Define some input parameters
        box = [0.1, 0.2, 0.3, 0.4]
        size = [100, 200]

        # Call the function
        result = mn2.get_most_probable_bounds(box, size)

        # Check that the result has the expected type and value
        expected_result = np.int32([20, 20, 60, 40])
        np.testing.assert_array_equal(result, expected_result)

    def test_get_most_probable_bounds_with_zero_box(self):
        # Define some input parameters
        box = [0, 0, 0, 0]
        size = [100, 200]

        # Call the function
        result = mn2.get_most_probable_bounds(box, size)

        # Check that the result has the expected type and value
        expected_result = np.int32([0, 0, 0, 0])
        np.testing.assert_array_equal(result, expected_result)

    def test_get_most_probable_bounds_with_large_box(self):
        # Define some input parameters
        box = [0.5, 0.5, 2.0, 2.0]
        size = [100, 200]

        # Call the function
        result = mn2.get_most_probable_bounds(box, size)

        # Check that the result has the expected type and value
        expected_result = np.int32([100, 50, 200, 100])
        np.testing.assert_array_equal(result, expected_result)  
        
if __name__ == '__main__':
    unittest.main()
