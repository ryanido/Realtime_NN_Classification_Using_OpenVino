import cv2
import numpy as numpy

def resize_image_letterbox(image, size, interpolation=cv2.INTER_LINEAR):
  ih, iw = image.shape[0:2]
  w, h = size
  scale = min(w / iw, h / ih)
  nw = int(iw * scale)
  nh = int(ih * scale)
  dx = (w - nw) // 2
  dy = (h - nh) // 2

  image = cv2.resize(image, (nw, nh), interpolation=interpolation)

  resized_image = numpy.pad(
    array           = image, 
    pad_width       = ((dy, dy + (h - nh) % 2), (dx, dx + (w - nw) % 2), (0, 0)),
    mode            = 'constant',
    constant_values = 0
  )

  return resized_image

def get_inference_input(image, input_size):
    image_size = [image.shape[0], image.shape[1]]
    image = resize_image_letterbox(image, [input_size, input_size], 2)
    data = numpy.array(image)
    data = numpy.transpose(data, (2, 0, 1)) # hwc -> chw
    data = data.reshape(1, 3, input_size, input_size) # 3,416,416 -> 1,3,416,416
    data = data.astype('float32')

    input_data = {
      'input_1': data,
      'image_shape': numpy.array([image_size], dtype=numpy.float32)
    }

    return input_data
