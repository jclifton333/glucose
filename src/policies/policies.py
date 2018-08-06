import numpy as np


def maximize_q_function():
  return


def model_based(env, transition_model_fitter):
  """

  :param env: Glucose instance
  :param transition_model_fitter:
  :return:
  """
  # Fit transition model
  transition_model = transition_model_fitter()
  X = env.get_state_history_as_array()
  X_next = np.vstack(env.S[2:])
  transition_model.fit(X, X_next)

  # Simulation optimization
  return


def fitted_q(env, gamma, regressor, number_of_value_iterations):
  """

  :param env:
  :param regressor:
  :param number_of_value_iterations:
  :return:
  """
  X = env.get_state_history_as_array()
  target = np.hstack(env.R)
  # Fit one-step q fn
  reg = regressor()
  reg.fit(X, target)

  # Fit longer-horizon q fns
  for k in range(number_of_value_iterations):
    q_max, optimal_action = maximize_q_function(reg, X)
    target += gamma * q_max
    reg.fit(X, target)

  return optimal_action


