"""
Functions for estimating bellman error using combinations of model-based and model-free temporal difference errors.

Refer to model based and model free BE estimators as delta_mb, delta_mf, respectively.

To get (estimated) MSE-optimal convex combination, we need to estimate
  - sampling variance of delta_mb and delta_mf
  - bias of delta_mb
  - sampling correlation of delta_mb and delta_mf
  - signal-to-noise ratios of resp sampling dbns
(refer to formula in http://interstat.statjournals.net/YEAR/2001/articles/0103002.pdf)
See also https://hal.archives-ouvertes.fr/hal-00936024/file/A-general-procedure-to-combine-estimators.pdf

---

As an alternative to combining using the kernel/bootstrap estimates of bias and variance (in order to apply the above
method), I'm also trying just choosing alpha to minimize ( \alpha mb_backup + (1 - \alpha) mf_backup - kernel_backup)**2,
where kernel_backup is a kde estimator of backups.
"""
import numpy as np
from policies.policies import expected_q_max, maximize_q_function_at_block
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel


def delta_mf_variance_pooled(delta_mf, bootstrap_weight_array):
  """
  Get a bootstrap estimator of the variance of delta_mf, pooling equally across all temporal differences rather than 
  treating each (S, A) pair differently.

  :param delta_mf: array of temporal differences
  :param bootstrap_weight_array: number_of_bootstrap_replicates x len(td) - size array of bootstrap multipliers
  :return:
  """
  n = len(delta_mf)
  bootstrapped_deltas = np.zeros((0, n))
  for multiplier in bootstrap_weight_array:
    delta_b = np.multiply(delta_mf, multiplier)
    bootstrapped_deltas = np.vstack((bootstrapped_deltas, delta_b))
  elementwise_variances = np.variance(bootstrapped_deltas, axis=0)
  return np.mean(elementwise_variances), bootstrapped_deltas


def delta_mb_bias_and_variance_pooled(delta_mb, q_fn, env, gamma, X, Sp1, transition_model_fitter,
                                      bootstrap_weight_array):
  """
  Estimate sampling variance of delta_mb by fitting model to bootstrapped samples of the data (and pooling).
  """
  n = X.shape[0]
  transition_model = transition_model_fitter()
  bootstrapped_deltas = np.zeros((0, n))
  for multiplier in bootstrap_weight_array:
    X_b = np.multiply(multiplier, X_b)
    transition_model_fitter.fit(X_b, Sp1)
    R_b = transition_model_fitter.expected_reward(X)
    expected_q_max_array, _ = expected_q_max(q_fn, X, env, transition_model)
    delta_b = R_b + gamma * expected_q_max_array - q_fn(X)
    bootstrapped_deltas = np.vstack((bootstrapped_deltas, delta_b))
  elementwise_variances = np.variance(bootstrapped_deltas, axis=0)
  elementwise_means = np.mean(bootstrapped_deltas, axis=0)
  return np.mean(elementwise_variances), bootstrapped_deltas, delta_mb - elementwise_means


def estimate_combination_weights(delta_mf, q_fn, env, gamma, X, Sp1, transition_model_fitter, alpha_mf_prev,
                                 number_of_bootstrap_samples=10):
  """

  :param delta_mf:
  :param q_fn:
  :param env:
  :param gamma:
  :param X:
  :param Sp1:
  :param transition_model_fitter:
  :param alpha_mf_prev: previous estimate of alpha_mf, for estimating bias and signal-to-noise
  :param number_of_bootstrap_samples:
  :return:
  """
  n = len(delta_mf)
  bootstrap_weight_array = np.random.exponential(size=(number_of_bootstrap_samples, n))

  # Bias and variance estimates
  var_delta_mf, bootstrapped_delta_mf = delta_mf_variance_pooled(delta_mf, bootstrap_weight_array)
  var_delta_mb, bootstrapped_delta_mb, bias_delta_mb =\
    delta_mb_bias_and_variance_pooled(delta_mb, q_fn, env, gamma, X, Sp1, transition_model_fitter,
                                      bootstrap_weight_array)

  # Correlation estimate
  elementwise_correlations = np.array([pearsonr(bootstrapped_delta_mf[:, i], bootstrapped_delta_mb[:, i])[0]
                                       for i in range(n)])
  rho = np.mean(elementwise_correlations)  # Pooling across (S, A) pairs

  # Estimate k_mb (where E[delta_mb] = k_mb * BE)
  delta_hat = delta_mb - bias_delta_mb
  k_mb = delta_mb / delta_hat
  k_mf = 1  # delta_mf is unbiased

  # Estimate signal-to-noise ratios v
  v_mf = var_delta_mf / delta_hat
  v_mb = var_delta_mb / delta_hat

  # Estimate lambda
  lambda_ = (k_mf**2 * v_mb) / (k_mb**2 * v_mf)

  # Estimate alphas
  alpha_mf = (lambda_ * (lambda_ - rho)) / (1 - 2*rho*lambda_ + lambda_**2 + (1 + rho**2)*(v_mf / k_mb))
  alpha_mb = 1 - alpha_mf

  # return {'var_delta_mf': var_delta_mf, 'var_delta_mb': var_delta_mb, 'correlation': correlation}
  return alpha_mb, alpha_mf


def mb_backup(q_fn, env, gamma, X, transition_model, reward_only=False):
  R_expected = transition_model.expected_glucose_reward_at_block(X, env)
  if not reward_only:
    expected_q_max_ = expected_q_max(q_fn, X, env, transition_model)
    backup = R_expected + gamma * expected_q_max_
  else:
    backup = R_expected
  return backup


def mf_backup(q_fn, env, gamma, Xp1, reward_only=False):
  R = np.hstack(env.R)
  if not reward_only
    q_max = maximize_q_function_at_block(q_fn, Xp1, env)
    backup = R + gamma * q_max
  else:
    backup = R
  return backup


def bootstrapped_kernel_mse_components(q_fn, q_mf_backup, env, gamma, transition_model, number_of_bootstrap_samples, X,
                                       Xp1, Sp1, reward_only):
  n = X.shape[0]
  bootstrap_mf_backup_dbn = np.zeros((0, n))
  bootstrap_mb_backup_dbn = np.zeros((0, n))

  pairwise_kernels_ = pairwise_kernels(X)
  kernel_sums = np.sum(pairwise_kernels, axis=1)
  pairwise_kernels_ = np.dot(np.diag(1 / kernel_sums), pairwise_kernels_)  # Weight kernels by normalizing constants

  mf_variances = np.zeros(n)
  mb_variances = np.zeros(n)
  covariances = np.zeros(n)

  for b in range(number_of_bootstrap_samples):
    # Compute backups
    multiplier = np.random.exponential(size=n)
    X_b, Sp1_b = np.multiply(multiplier, X_b), np.multiply(multiplier, Sp1)
    Xp1_b, R_b = np.multiply(multiplier, Xp1), np.multiply(multiplier, np.hstack(env.R))
    transition_model.fit(X_b, Sp1_b)
    q_mb_backup_b = mb_backup(q_fn, env, gamma, X_b, transition_model, reward_only=reward_only)
    q_mf_backup_b = mf_backup(q_fn, env, gamma, Xp1_b, reward_only=reward_only)
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

  correlations = covariances / np.sqrt(np.multiply(mb_variances, mf_variances))
  return {'mf_variances': mf_variances, 'mb_biases': mb_biases, 'mb_variances': mb_variances,
          'correlations': correlations}


def kernel_bias_estimator(q_fn, env, gamma, transition_model, X, Xp1, R, kernel, reward_only):
  n = X.shape[0]
  q_mb_backup = mb_backup(q_fn, env, gamma, X, transition_model, reward_only=reward_only)
  q_mf_backup = mf_backup(q_fn, env, gamma, R, Xp1, reward_only=reward_only)

  mb_biases = []
  # Compare mb backups to (kernel-smoothed) mf backups
  for t in range(n):
    x = X[t, :]
    backup_at_x = q_mb_backup[t]
    kernel_bias_estimator_ = 0.0
    kernel_normalizing_constant = 0.0
    for s in range(n):
      x_s = X[s, :]
      mf_backup_at_x_s = q_mf_backup[s]
      kernel_distance = kernel(x, x_s)
      kernel_bias_estimator_ += (backup_at_x - mf_backup_at_x_s) * kernel_distance
      kernel_normalizing_constant += kernel_distance
    mb_biases.append(kernel_bias_estimator_ / kernel_normalizing_constant)
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


# ToDo: Split into model-smoothed reward and model-smoothed expected_q_max
def model_smoothed_backup(q_fn, env, gamma, X, Sp1, transition_model, kernel=rbf_kernel, reward_only=False,
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

  # Get backed_up_q_mf
  X, Xp1 = X[:-1, :], X[1:, :]

  q_mf_backup = mf_backup(q_fn, env, gamma, Xp1, reward_only=reward_only)
  q_mb_backup = mb_backup(q_fn, env, gamma, X, transition_model, reward_only=reward_only)

  # Get components needed to estimate alphas
  bootstrapped_mse_components_ = bootstrapped_kernel_mse_components(q_fn, env, gamma, transition_model,
                                                                    number_of_bootstrap_samples, X, Xp1, Sp1,
                                                                    reward_only)
  mb_biases_ = kernel_bias_estimator(q_fn, env, gamma, transition_model, X, Xp1, R, kernel, reward_only)

  alpha_mb, alpha_mf = \
    estimate_weights_from_mse_components(q_mb_backup, mb_biases_,
                                         bootstrapped_mse_components_['mb_variances'],
                                         bootstrapped_mse_components_['mf_variances'],
                                         bootstrapped_mse_components_['correlations'])

  return alpha_mb*q_mb_backup + alpha_mf*q_mf_backup


def optimal_convex_combination(x1, x2, y):
  """
  Choose \alpha to minimize || \alpha x1 + (1 - \alpha) x2 - y ||^2.


  :param x1:
  :param x2:
  :param y:
  :return:
  """
  x1_minus_x2 = x1 - x2
  alpha = np.dot(x1_minus_x2, y - x2) / np.dot(x1_minus_x2, x1_minus_x2)
  if alpha < 0:
    alpha = 0.0
  elif alpha > 1:
    alpha = 1.0
  return alpha


def model_smoothed_reward(env, X, transition_model, pairwise_kernels_):
  """
  Estimated MSE-optimal combo of model-free and model-based conditional reward estimates.

  :param env:
  :param gamma:
  :param X:
  :param transition_model:
  :param pairwise_kernels_: matrix of pairwise kernels for the observed x's; this is updated online during the episode
  :param kernel:
  :param number_of_bootstrap_samples:
  :return:
  """
  # mb and mf reward estimates
  r_mb = transition_model.expected_glucose_reward_at_block(X, env)
  r_mf = np.hstack(env.R)

  # kde reward estimate
  r_kde = np.dot(pairwise_kernels_, r_mf)

  # get optimal mixing weight
  alpha_mb = optimal_convex_combination(r_mb, r_mf, r_kde)
  alpha_mf = 1 - alpha_mb

  return alpha_mb*r_mb + alpha_mf*r_mf, r_mb, r_mf, r_kde


def model_smoothed_qmax(q_fn, q_mf_backup, q_mb_backup, q_kde_backup, env, gamma, X, transition_model,
                        pairwise_kernels_):

  # backup with mb and mf E[q_max] estimates
  q_max_mb = expected_q_max(q_fn, X, env, transition_model)
  q_max_mf, _ = maximize_q_function_at_block(q_fn, Xp1, env)
  q_mb_backup += gamma * q_max_mb
  q_mf_backup += gamma * q_max_mf

  # backup kde estimate
  q_kde_backup += gamma * np.dot(pairwise_kernels_, q_max_mf)

  # get optimal mixing weight
  alpha_mb = optimal_convex_combination(q_mb_backup, q_mf_backup, q_kde_backup)
  alpha_mf = 1 - alpha_mb

  return alpha_mb*q_mb_backup + alpha_mf*q_mf_backup




