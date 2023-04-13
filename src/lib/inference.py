from openvino.inference_engine import IECore


class Inference:

  _request_slot = 0

  def __init__(
      self,
      path_to_xml: str,
      path_to_bin: str,
      device_name: str,
      num_requests: int = 1
  ):
    self.path_to_xml = path_to_xml
    self.path_to_bin = path_to_bin
    self.device_name = device_name
    self.num_requests = num_requests

    ie = IECore()
    self.network = ie.read_network(self.path_to_xml, self.path_to_bin)
    self.exec_network = ie.load_network(
        network=self.network,
        device_name=self.device_name,
        num_requests=self.num_requests)

  def infer(self, input: any):
    result = self.exec_network.infer(input)
    return result

  def async_infer(self, input: any, callback):
    def _callback(status, request_slot):
      if status == 0:
        callback(request=self.exec_network.requests[request_slot])

    self.exec_network.requests[self._request_slot].wait()
    self.exec_network.requests[self._request_slot].async_infer(input)
    self.exec_network.requests[self._request_slot].set_completion_callback(
        py_callback=_callback,
        py_data=self._request_slot)

    self._request_slot = (self._request_slot + 1) % self.num_requests
