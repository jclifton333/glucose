import pdb
import numpy as np
import bellman_error_estimation as be
from helpers import maximize_q_function_at_block, update_pairwise_kernels_
from sklearn.metrics.pairwise import rbf_kernel


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
    q_max, _ = maximize_q_function_at_block(reg.predict, Xp1, env)
    target += gamma * q_max
    reg.fit(X, target)

  # Maximize final q iterate to get next action
  _, list_of_optimal_actions = maximize_q_function_at_block(reg.predict, X, env)

  # Last entry of list gives optimal action at current state
  optimal_action = list_of_optimal_actions[-1]
  return optimal_action


def model_smoothed_fitted_q(env, gamma, regressor, number_of_value_iterations, transition_model_fitter,
                            pairwise_kernels_, kernel_sums, kernel=rbf_kernel, smoothing_method='kde'):
  """

  :param env:
  :param gamma:
  :param regressor:
  :param number_of_value_iterations:
  :param transition_model_fitter:
  :param pairwise_kernels_:
  :param kernel_sums:
  :param kernel:
  :param smoothing_method: either 'kde' or 'mse', for two different ways of averaging mb and mf backup estimates.
  :return:
  """
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  transition_model = transition_model_fitter()
  transition_model.fit(X, Sp1)

  X = env.get_state_history_as_array()
  X, Xp1 = X[:-1, :], X[1:, :]

  pairwise_kernels_, kernel_sums = update_pairwise_kernels_(pairwise_kernels_, kernel, kernel_sums, X)
  averaged_backup, mb_backup, mf_backup, kde_backup = be.model_smoothed_reward(env, transition_model,
                                                                               pairwise_kernels_,
                                                                               method=smoothing_method)

  # Fit one-step q fn
  reg = regressor()
  reg.fit(X, averaged_backup)

  # Fit longer-horizon q fns
  for k in range(number_of_value_iterations):
    averaged_backup, mb_backup, mf_backup, kde_backup, alpha_mb = \
      be.model_smoothed_qmax(reg.predict, mb_backup, mf_backup, kde_backup, env, gamma, X, Xp1, transition_model,
                             pairwise_kernels_, method=smoothing_method)
    reg.fit(X, averaged_backup)
  # Maximize final q iterate to get next action
  _, list_of_optimal_actions = maximize_q_function_at_block(reg.predict, X, env)

  # Last entry of list gives optimal action at current state
  optimal_action = list_of_optimal_actions[-1]
  return optimal_action, pairwise_kernels_, kernel_sums






