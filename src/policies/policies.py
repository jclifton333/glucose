import numpy as np


def maximize_q_function_at_x(q_fn, x, env):
  """

  :param q_fn: Function from feature vectors x to q-values.
  :param x: Feature vector from Glucose environment at which to maximize Q.
  :param env: Glucose environment
  :return:
  """
  x0, x1 = env.get_state_at_action(0, x), env.get_state_at_action(1, x)
  q0, q1 = q_fn(x0), q_fn(x1)
  return np.max([q0, q1]), np.armgax([q0, q1])


def maximize_q_function_at_block(q_fn, X, env):
  """

  :param q_fn:
  :param X: Nx9 array of Glucose features.
  :param env:
  :return:
  """
  array_of_q_maxes = []
  list_of_optimal_actions = []
  for x in X:
    q_max, optimal_action = maximize_q_function_at_x(q_fn, x, env)
    array_of_q_maxes.append(q_max)
    list_of_optimal_actions.append(optimal_action)
  return np.array(array_of_q_maxes), list_of_optimal_actions


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
    q_max_array, _ = maximize_q_function(reg, X, env)
    target += gamma * q_max
    reg.fit(X, target)

  # Maximize final q iterate to get next action
  _, list_of_optimal_actions = maximize_q_function(reg, X, env)

  # Last entry of list gives optimal action at current state
  optimal_action = list_of_optimal_actions[-1]
  return optimal_action


