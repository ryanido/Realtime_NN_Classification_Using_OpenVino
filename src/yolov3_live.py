import cv2
from inference import Inference
from put_inference_output import put_inference_output
import time

def yolov3(deviceName) :
  total_inference_frames = 0
  start_time = time.time()
  fps_str = ''
  inference = Inference(
    device=deviceName, 
    xml_file='../model/yolo-v3-onnx/yolo-v3-onnx.xml', 
    bin_file='../model/yolo-v3-onnx/yolo-v3-onnx.bin', 
    input_size=416
  )

  video = cv2.VideoCapture(0)
  while (True):
    ret, frame = video.read()

    inference_output = inference.infer(frame)

    put_inference_output(frame, inference_output)

    total_inference_frames += 1

    if total_inference_frames == 5:
      end_time = time.time()
      fps_str = f'{(5 / (end_time - start_time)):2.1f} FPS'

    cv2.putText(frame,fps_str, 
        (frame.shape[1] - 150,frame.shape[0] - 50), 
        cv2.FONT_HERSHEY_PLAIN, 
        1.8,
        (0,255,0),
        1,
        2)

    cv2.imshow("YoloV3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  video.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  yolov3()
