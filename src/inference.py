from openvino.inference_engine import IECore
from get_inference_input import get_inference_input

class Inference:
  def __init__(self, device: str, xml_file: str, bin_file: str, input_size: int):
    ie = IECore()

    if(device in ie.available_devices):
      self.device = device
    else:
      self.device = ie.available_devices[0]

    network = ie.read_network(model=xml_file, weights=bin_file)
    self.exec_network = ie.load_network(network, self.device)
    self.input_size = input_size

  def infer(self, image):
    input_data = get_inference_input(image, self.input_size)
    result = self.exec_network.infer(input_data)

    return result
