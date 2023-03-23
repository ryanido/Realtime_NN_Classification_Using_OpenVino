import queue
import time
import cv2
from inference import Inference
from get_inference_input import get_inference_input
from put_inference_output import put_inference_output

video = cv2.VideoCapture(0)

inference = Inference(
    path_to_xml='./models/ssdlite-mobilenet-v2/ssdlite-mobilenet-v2.xml',
    path_to_bin='./models/ssdlite-mobilenet-v2/ssdlite-mobilenet-v2.bin',
    device_name='CPU',
    num_requests=4
)


inference = Inference(
    path_to_xml='./models/yolo/yolo.xml',
    path_to_bin='./models/yolo/yolo.bin',
    device_name='CPU',
    num_requests=4
)

display_queue = queue.Queue()

fps_str = '? FPS'
total_frames = 0
start_time = time.time()
end_time = time.time()


def get_callback(frame):
  def callback(request):
    inference_output = request.output_blobs['DetectionOutput'].buffer[0][0]
    put_inference_output(frame, inference_output)
    display_queue.put(frame)

  return callback


while (True):
  ret, frame = video.read()

  if (total_frames >= 30):
    end_time = time.time()
    fps_str = f'{(30 / (end_time - start_time)):2.1f} FPS'
    total_frames = 0
    start_time = time.time()

  if not display_queue.empty():
    current_frame = display_queue.get()

    cv2.putText(current_frame, fps_str,
                (current_frame.shape[1] - 150, current_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_PLAIN,
                1.8,
                (0, 255, 0),
                1,
                2)

    cv2.imshow('SSDLite Mobilenet V2', current_frame)

    total_frames += 1

  inference_input = get_inference_input(frame, 300, 'image_tensor')
  inference.async_infer(inference_input, get_callback(frame))

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()
