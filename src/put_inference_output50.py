import numpy
import cv2

import lib.coco80_labels as coco
import lib.colour_palette as palette


def get_most_probable_bounds(box, size):
  box[0] = max(int(box[0]), 0)  # ymin
  box[1] = max(int(box[1]), 0)  # xmin
  box[2] = min(int(box[2]), size[0])  # ymax
  box[3] = min(int(box[3]), size[1])  # xmax

  return numpy.int32(box)


def put_inference_output(image, inference_output):
  boxes = inference_output['yolonms_layer_1/ExpandDims_1:0'][0]
  scores = inference_output['yolonms_layer_1/ExpandDims_3:0'][0]
  indices = inference_output['yolonms_layer_1/concat_2:0']

  colours = palette.get_colour_palette(len(coco.labels))
  confidence_threshold = 0.5
  label_height = 18

  for index in indices:
    if (index[0] == -1):
      break

    score = scores[tuple(index[1:])]

    if (score >= confidence_threshold):
      # ymin, xmin, ymax, xmax
      box = boxes[index[2]]
      box = get_most_probable_bounds(box, [image.shape[0], image.shape[1]])
      label_index = index[1]
      ymin = box[0]
      xmin = box[1]
      ymax = box[2]
      xmax = box[3]

      det_label = f'{score*100:2.1f}% {coco.labels[label_index]}'

      colour = colours[label_index]
      text_colour = (28, 28, 28)

      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colour, 2)
      cv2.rectangle(image, (xmin, ymin - label_height), (xmax, ymin),
                    colour, -1)
      cv2.rectangle(image, (xmin, ymin - label_height), (xmax, ymin),
                    colour, 2)
      cv2.putText(image, det_label, (xmin, ymin - 7),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_colour, 1)
