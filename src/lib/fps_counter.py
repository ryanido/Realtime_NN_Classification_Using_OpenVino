import time


class FPSCounter:

  fps = 0
  _frame = 0
  _elapsed = time.time()

  def __init__(self, target: int = 1):
    self.target = target

  # Report a new frame
  def frame(self):
    self._frame += 1

    if self._frame == self.target:
      now = time.time()
      self.fps = self.target / (now - self._elapsed)
      self._elapsed = now
      self._frame = 0

  # Get the FPS number
  def get(self):
    return self.fps
