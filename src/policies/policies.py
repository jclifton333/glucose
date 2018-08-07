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


def fitted_q(env, gamma, regressor, number_of_value_iterations):
  X = env.get_state_history_as_array()
  X, Xp1 = X[:-1, :], X[1:, :]
  target = np.hstack(env.R)
  # Fit one-step q fn
  reg = regressor()
  reg.fit(X, target)

  # Fit longer-horizon q fns
  for k in range(number_of_value_iterations):
    q_max_array, _ = maximize_q_function_at_block(reg, Xp1, env)
    target += gamma * q_max
    reg.fit(X, target)

  # Maximize final q iterate to get next action
  _, list_of_optimal_actions = maximize_q_function_at_block(reg, X, env)

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


# ToDO: This and fitted_q don't need to be different functions
def model_enriched_fitted_q(env, gamma, q_regressor, transition_model_fitter, number_of_value_iterations):
  """
  Fitted Q iteration, with observed data supplemented by expectations from estimated transition model.
  "mb" refers to "model-based".

  :param env:
  :param gamma:
  :param q_regressor: Model constructor for fitted q iteration.
  :param transition_model_fitter:
  :param number_of_value_iterations:
  :return:
  """

  # Get observed data
  X_obs = env.get_state_history_as_array()
  X_obs, Xp1_obs = X_obs[:-1, :], X_obsp1[1:, :]
  R_obs = np.hstack(env.R)

  # Get simulated data
  transition_model = transition_model_fitter()
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  transition_model.fit(X, Sp1)

  # Combine
  X_combined = np.vstack((X_obs, X_obs))
  Sp1_mb = transition_model.conditional_expectation_at_block(X_obs)
  R_mb = np.array([env.reward_function(Sp1_mb[t, :], Sp1_mb[t+1, :]) for t in range(Sp1_mb.shape[0] - 1)])
  target = np.hstack((R_obs, R_mb))

  # Fit one-step q fn
  reg = q_regressor()
  reg.fit(X_combined, target)

  # Fit longer-horizon q fns
  for k in range(number_of_value_iterations):
    q_max_array_at_obs, _ = maximize_q_function_at_block(reg, X_obs_p1, env)
    expected_q_max_array, _ = expected_q_max(reg, X_obs, env, transition_model)
    target += gamma * q_max
    reg.fit(X, target)

  # Maximize final q iterate to get next action
  _, list_of_optimal_actions = maximize_q_function_at_block(reg, X, env)

  # Last entry of list gives optimal action at current state
  optimal_action = list_of_optimal_actions[-1]

  return optimal_action





