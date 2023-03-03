import cv2
from inference import Inference
from put_inference_output import put_inference_output

video = cv2.VideoCapture(0)

inference = Inference(
  device='CPU', 
  xml_file='../model/yolo-v3-onnx/yolo-v3-onnx.xml', 
  bin_file='../model/yolo-v3-onnx/yolo-v3-onnx.bin', 
  input_size=416
)

while (True):
  ret, frame = video.read()

  inference_output = inference.infer(frame)

  put_inference_output(frame, inference_output)

  cv2.imshow("frame", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()
