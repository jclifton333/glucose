import numpy as np
import pdb
from src.policies.helpers import expected_q_max, maximize_q_function_at_block


def mb_backup(q_fn, env, gamma, X, transition_model):
  r_expected = transition_model.expected_glucose_reward_at_block(X, env)
  expected_q_max_ = expected_q_max(q_fn, X, env, transition_model)
  backup = r_expected + gamma * expected_q_max_
  return backup


def mf_backup(q_fn, env, gamma, Xp1):
  r = np.hstack(env.R)
  q_max, _ = maximize_q_function_at_block(q_fn, Xp1, env)
  backup = r + gamma * q_max
  return backup


def mse_components_for_reward(r_mf, r_mb, env, transition_model, number_of_bootstrap_samples, X, Sp1, pairwise_kernels_):
  n = X.shape[0]

  mf_variances = np.zeros(n)
  mb_variances = np.zeros(n)
  covariances = np.zeros(n)

  bootstrap_r_mf_dbn = np.zeros((0, n))
  bootstrap_r_mb_dbn = np.zeros((0, n))

  for b in range(number_of_bootstrap_samples):
    # Compute backups
    multiplier = np.random.exponential(size=n)
    X_b, Sp1_b, r_mf_b = np.multiply(X, multiplier), np.multiply(Sp1, multiplier), np.multiply(r_mf, multiplier)
    transition_model.fit(X_b, Sp1_b)
    r_mb_b = transition_model.expected_glucose_reward_at_block(X, env)
    bootstrap_r_mf_dbn = np.vstack((bootstrap_r_mf_dbn, r_mf_b))
    bootstrap_r_mb_dbn = np.vstack((bootstrap_r_mb_dbn, r_mb_b))

    # Update variance_estimates
    mf_squared_error_b = (r_mf_b - r_mf)**2
    mf_squared_error_b_kernelized = np.dot(pairwise_kernels_, mf_squared_error_b)
    mf_variances += mf_squared_error_b_kernelized / number_of_bootstrap_samples
    mb_squared_error_b = (r_mb_b - r_mb)**2
    mb_squared_error_b_kernelized = np.dot(pairwise_kernels_, mb_squared_error_b)
    mb_variances += mb_squared_error_b_kernelized / number_of_bootstrap_samples

    # Update covariance estimates
    covariances_b = np.multiply(r_mf_b - r_mf, r_mb_b - r_mb)
    covariances_b_kernelized = np.dot(pairwise_kernels_, covariances_b)
    covariances += covariances_b_kernelized

  correlations = covariances / np.sqrt(np.multiply(mb_variances, mf_variances))
  return {'mf_variances': mf_variances, 'mb_variances': mb_variances,
          'correlations': correlations}


def mse_components_for_qmax(q_fn, q_mb_backup, q_mf_backup, env, gamma, transition_model, number_of_bootstrap_samples,
                            X, Xp1, Sp1, pairwise_kernels_):
  n = X.shape[0]

  mf_variances = np.zeros(n)
  mb_variances = np.zeros(n)
  covariances = np.zeros(n)

  bootstrap_mb_backup_dbn = np.zeros((0, n))
  bootstrap_mf_backup_dbn = np.zeros((0, n))

  for b in range(number_of_bootstrap_samples):
    # Compute backups
    multiplier = np.random.exponential(size=n)
    X_b, Sp1_b = np.multiply(multiplier, X), np.multiply(multiplier, Sp1)
    Xp1_b, R_b = np.multiply(multiplier, Xp1), np.multiply(multiplier, np.hstack(env.R))
    transition_model.fit(X_b, Sp1_b)
    q_mb_backup_b = mb_backup(q_fn, env, gamma, X_b, transition_model)
    q_mf_backup_b = mf_backup(q_fn, env, gamma, Xp1_b)
    bootstrap_mf_backup_dbn = np.vstack((bootstrap_mf_backup_dbn, q_mf_backup_b))
    bootstrap_mb_backup_dbn = np.vstack((bootstrap_mb_backup_dbn, q_mb_backup_b))

    # Update variance_estimates
    mf_squared_error_b = (q_mf_backup_b - q_mf_backup)**2
    mf_squared_error_b_kernelized = np.dot(pairwise_kernels_, mf_squared_error_b)
    mf_variances += mf_squared_error_b_kernelized / number_of_bootstrap_samples
    mb_squared_error_b = (q_mb_backup_b - q_mb_backup)**2
    mb_squared_error_b_kernelized = np.dot(pairwise_kernels_, mb_squared_error_b)
    mb_variances += mb_squared_error_b_kernelized / number_of_bootstrap_samples

    # Update covariance estimates
    covariances_b = np.multiply(q_mf_backup_b - q_mf_backup, q_mb_backup_b - q_mb_backup)
    covariances_b_kernelized = np.dot(pairwise_kernels_, covariances_b)
    covariances += covariances_b_kernelized

  mf_variances = np.max([mf_variances, np.ones(n)*0.1], axis=0)
  correlations = covariances / np.sqrt(np.multiply(mb_variances, mf_variances))
  return {'mf_variances': mf_variances, 'mb_variances': mb_variances,
          'correlations': correlations}


def kernel_bias_estimator(q_mb_backup, q_mf_backup, pairwise_kernels_):

  # Compare mb backups to (kernel-smoothed) mf backups
  mb_biases = np.dot(pairwise_kernels_, q_mb_backup - q_mf_backup)

  return {'mb_biases': np.array(mb_biases)}


def estimate_weights_from_mse_components(q_mb_backup, mb_biases, mb_variances, mf_variances, correlations):
  # ToDo: Think about how to get preliminary estimate of q_backup
  q_backup_hat = q_mb_backup - mb_biases

  # Estimate bias coefficients k
  k_mb = q_mb_backup / q_backup_hat
  k_mf = 1  # q_mf_backup is unbiased

  # Estimate signal-to-noise ratios v
  v_mf = mf_variances / q_backup_hat
  v_mb = mb_variances / q_backup_hat

  # Estimate lambda
  lambda_ = (k_mf ** 2 * v_mb) / (k_mb ** 2 * v_mf)

  # Estimate alphas
  alpha_mf = \
    (lambda_ * (lambda_ - correlations)) / (1 - 2 * correlations * lambda_ + lambda_ ** 2 + (1 + correlations ** 2) * (v_mf / k_mb))
  alpha_mb = 1 - alpha_mf
  return alpha_mb, alpha_mf


def model_smoothed_reward_using_mse(r_mb, r_mf, env, X, Sp1, transition_model, pairwise_kernels_,
                                    number_of_bootstrap_samples=100):
  """
  Estimate Bellman backup of q_fn by taking estimated MSE-optimal combination of model-free and model-based backups.

  :param q_fn:
  :param env:
  :param gamma:
  :param X:
  :param Sp1:
  :param transition_model:
  :param kernel: Function that takes two arrays and returns kernel distance, e.g.
                 sklearn.metrics.pairwise.rbf_kernel
  :param reward_only: bool for only returning estimate of E[R] instead of E[R + \gamma max Q]
  :param number_of_bootstrap_samples: Bootstrapping is used to estimate MSE-minimizing weights.
  :return:
  """
  # Get components needed to estimate alphas
  bootstrapped_mse_components_ = \
    mse_components_for_reward(r_mf, r_mb, env, transition_model, number_of_bootstrap_samples, X, Sp1, pairwise_kernels_)

  mb_biases_ = kernel_bias_estimator(r_mb, r_mf, pairwise_kernels_)

  alpha_mb, alpha_mf = estimate_weights_from_mse_components(r_mb, mb_biases_['mb_biases'],
                                                             bootstrapped_mse_components_['mb_variances'],
                                                             bootstrapped_mse_components_['mf_variances'],
                                                             bootstrapped_mse_components_['correlations'])

  return alpha_mb, alpha_mf


def model_smoothed_qmax_using_mse(q_mb_backup, q_mf_backup, q_fn, env, gamma, X, Xp1, Sp1, transition_model,
                                  pairwise_kernels_, number_of_bootstrap_samples=100):
  """
  Estimate Bellman backup of q_fn by taking estimated MSE-optimal combination of model-free and model-based backups.

  :param q_fn:
  :param env:
  :param gamma:
  :param X:
  :param Sp1:
  :param transition_model:
  :param kernel: Function that takes two arrays and returns kernel distance, e.g.
                 sklearn.metrics.pairwise.rbf_kernel
  :param reward_only: bool for only returning estimate of E[R] instead of E[R + \gamma max Q]
  :param number_of_bootstrap_samples: Bootstrapping is used to estimate MSE-minimizing weights.
  :return:
  """

  # Get components needed to estimate alphas
  bootstrapped_mse_components_ = \
    mse_components_for_qmax(q_fn, q_mb_backup, q_mf_backup, env, gamma, transition_model, number_of_bootstrap_samples, X,
                             Xp1, Sp1, pairwise_kernels_)

  mb_biases_ = kernel_bias_estimator(q_mb_backup, q_mf_backup, pairwise_kernels_)

  alpha_mb, alpha_mf = estimate_weights_from_mse_components(q_mb_backup, mb_biases_['mb_biases'],
                                                             bootstrapped_mse_components_['mb_variances'],
                                                             bootstrapped_mse_components_['mf_variances'],
                                                             bootstrapped_mse_components_['correlations'])

  return alpha_mb, alpha_mf
