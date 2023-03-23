import cv2
import numpy

numpy.set_printoptions(suppress=True, precision=3)


def get_inference_input(image, input_size: int, input_name: str):
  data = cv2.resize(image, (input_size, input_size))
  data = numpy.array(data)
  data = numpy.transpose(data, (2, 0, 1))  # hwc -> chw
  data = data.reshape(1, 3, input_size, input_size)
  data = data.astype('float32')

  input_data = {
      input_name: data
  }

  return input_data
