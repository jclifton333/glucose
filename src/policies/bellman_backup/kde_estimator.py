import numpy as np


def optimal_convex_combination(x1, x2, y):
  """
  Choose \alpha to minimize || \alpha x1 + (1 - \alpha) x2 - y ||^2.


  :param x1:
  :param x2:
  :param y:
  :return:
  """
  if np.array_equal(x1, x2):
    return 1.0
  else:
    x1_minus_x2 = x1 - x2
    alpha = np.dot(x1_minus_x2, y - x2) / np.dot(x1_minus_x2, x1_minus_x2)
    if alpha < 0:
      alpha = 0.0
    elif alpha > 1:
      alpha = 1.0
  return alpha