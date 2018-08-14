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
import kde_estimator as kde
import mse_estimator as mse
from src.policies.helpers import expected_q_max, maximize_q_function_at_block
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel
import pdb


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


def model_smoothed_reward(env, transition_model, pairwise_kernels_, method='kde', number_of_bootstrap_samples=100):
  """
  Estimated MSE-optimal combo of model-free and model-based conditional reward estimates.

  :param env:
  :param gamma:
  :param transition_model:
  :param pairwise_kernels_: matrix of pairwise kernels for the observed x's; this is updated online during the episode
  :param method: 'kde' or 'mse'
  :param number_of_bootstrap_samples:
  :return:
  """
  # mb and mf reward estimates
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  r_mb = transition_model.expected_glucose_reward_at_block(X, env)
  r_mf = np.hstack(env.R)

  if method == 'kde':
    r_kde = np.dot(pairwise_kernels_, r_mf)
    alpha_mb = kde.optimal_convex_combination(r_mb, r_mf, r_kde)
    alpha_mf = 1 - alpha_mb
  elif method == 'mse':
    alpha_mb, alpha_mf, bootstrapped_mse_components_ = \
      mse.model_smoothed_reward_using_mse(r_mb, r_mf, env, X, Sp1, transition_model,
                                          pairwise_kernels_, number_of_bootstrap_samples=number_of_bootstrap_samples)
    r_kde = None
  return alpha_mb*r_mb + alpha_mf*r_mf, r_mb, r_mf, r_kde, bootstrapped_mse_components_


def model_smoothed_qmax(q_fn, q_mf_backup, q_mb_backup, q_kde_backup, env, gamma, X, Xp1, Sp1, transition_model,
                        pairwise_kernels_, method='kde', number_of_bootstrap_samples=100):
  """

  :param q_fn:
  :param q_mf_backup:
  :param q_mb_backup:
  :param q_kde_backup:
  :param env:
  :param gamma:
  :param X:
  :param Xp1:
  :param Sp1:
  :param transition_model:
  :param pairwise_kernels_:
  :param method: 'kde' or 'mse'
  :return:
  """

  # backup with mb and mf E[q_max] estimates
  q_max_mb = expected_q_max(q_fn, X, env, transition_model)
  q_max_mf, _ = maximize_q_function_at_block(q_fn, Xp1, env)
  q_mb_backup += gamma * q_max_mb
  q_mf_backup += gamma * q_max_mf

  if method == 'kde':
    q_kde_backup += gamma * np.dot(pairwise_kernels_, q_max_mf)
    alpha_mb = kde.optimal_convex_combination(q_mb_backup, q_mf_backup, q_kde_backup)
    alpha_mf = 1 - alpha_mb
  elif method == 'mse':
    alpha_mb, alpha_mf, bootstrapped_mse_components_ = \
      mse.model_smoothed_qmax_using_mse(q_mb_backup, q_mf_backup, q_fn, env, gamma, X, Xp1, Sp1,
                                        transition_model, pairwise_kernels_,
                                        number_of_bootstrap_samples=number_of_bootstrap_samples)

  return alpha_mb*q_mb_backup + alpha_mf*q_mf_backup, q_mb_backup, q_mf_backup, q_kde_backup, alpha_mb, \
         bootstrapped_mse_components_




