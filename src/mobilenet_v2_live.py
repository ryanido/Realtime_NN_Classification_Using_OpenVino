import os
import queue
import cv2
from lib.inference import Inference
from lib.fps_counter import FPSCounter
from mobilenet_v2_live_put_inference_output import put_inference_output
from mobilenet_v2_live_get_inference_input import get_inference_input


basepath = os.path.dirname(os.path.realpath(__file__))
path_to_xml = f'{basepath}/../models/ssdlite-mobilenet-v2/ssdlite-mobilenet-v2.xml'
path_to_bin = f'{basepath}/../models/ssdlite-mobilenet-v2/ssdlite-mobilenet-v2.bin'


def mobilenet_v2_live(device_name='CPU', num_requests=4):

  video = cv2.VideoCapture(0)
  fps_counter = FPSCounter(target=30)
  inference = Inference(path_to_xml, path_to_bin, device_name, num_requests)
  output_queue = queue.Queue()

  # Get inference completion callback
  def bind_callback(frame, output_queue):
    def callback(request):
      inference_output = request.output_blobs['DetectionOutput'].buffer[0][0]
      put_inference_output(frame, inference_output)
      output_queue.put(frame)

    return callback

  while (True):
    ret, frame = video.read()

    # Put FPS and display frame
    if not output_queue.empty():
      output_frame = output_queue.get()

      cv2.putText(output_frame,
                  f'{fps_counter.get():.0f} FPS',
                  (output_frame.shape[1] - 90, output_frame.shape[0] - 28),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 127), 1, 2)

      cv2.imshow('SSDLite Mobilenet V2', output_frame)

      fps_counter.frame()

    # Infer current frame
    inference_input = get_inference_input(frame, 300, 'image_tensor')
    inference.async_infer(inference_input, bind_callback(frame, output_queue))

    # Stop loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  video.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  mobilenet_v2_live()
