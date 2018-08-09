import numpy as np
from bellman_error_estimation import model_smoothed_backup


def maximize_q_function_at_x(q_fn, x, env):
  """

  :param q_fn: Function from feature vectors x to q-values.
  :param x: Feature vector from Glucose environment at which to maximize Q.
  :param env: Glucose environment
  :return:
  """
  x0, x1 = env.get_state_at_action(0, x), env.get_state_at_action(1, x)
  q0, q1 = q_fn(x0.reshape(1, -1)), q_fn(x1.reshape(1, -1))
  return np.max([q0, q1]), np.argmax([q0, q1])


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


# def model_based(env, transition_model_fitter):
#   """
#
#   :param env: Glucose instance
#   :param transition_model_fitter:
#   :return:
#   """
#   # Fit transition model
#   transition_model = transition_model_fitter()
#   X, Sp1 = env.get_state_transitions_as_x_y_pair()
#   transition_model.fit(X, Sp1)
#
#   # Simulation optimization
#   return


def fitted_q(env, gamma, regressor, number_of_value_iterations, transition_model=None, model_smoothed_backup_=False):
  X = env.get_state_history_as_array()
  X, Xp1 = X[:-1, :], X[1:, :]
  target = np.hstack(env.R)
  # Fit one-step q fn
  reg = regressor()
  reg.fit(X, target)

  # Fit longer-horizon q fns
  for k in range(number_of_value_iterations):
    if model_smoothed_backup_:
      backup = model_smoothed_backup(q_fn, env, gamma, X, Sp1, transition_model)
    else:
      backup, _ = maximize_q_function_at_block(reg.predict, Xp1, env)
    target += gamma * backup
    reg.fit(X, target)

  # Maximize final q iterate to get next action
  _, list_of_optimal_actions = maximize_q_function_at_block(reg.predict, X, env)

  # Last entry of list gives optimal action at current state
  optimal_action = list_of_optimal_actions[-1]
  return optimal_action


def expected_q_max(q_fn, X, env, transition_model, num_draws=10):
  """
  Estimate the expected max of q_fn at Xp1 | X under estimated transition_model.

  :param q_fn:
  :param X:
  :param env:
  :param transition_model: A fit TransitionDensityEstimator object.
  :param num_draws:
  :return:
  """
  expected_q_max_ = np.zeros(X.shape[0])
  for draw in num_draws:
    Xp1 = transition_model.simulate_from_fit_model_at_block(X)
    q_max_at_draw = maximize_q_function_at_block(q_fn, Xp1, env)
    expected_q_max_ += (q_max_at_draw - expected_q_max_) / (draw + 1)
  return expected_q_max_



