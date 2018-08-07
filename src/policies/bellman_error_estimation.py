"""
Functions for estimating bellman error using combinations of model-based and model-free temporal difference errors.

Refer to model based and model free BE estimators as delta_mb, delta_mf, respectively.

To get (estimated) MSE-optimal convex combination, we need to estimate
  - sampling variance of delta_mb and delta_mf
  - sampling correlation of delta_mb and delta_mf
  - signal-to-noise ratios of resp sampling dbns
(refer to formula in http://interstat.statjournals.net/YEAR/2001/articles/0103002.pdf)
See also https://hal.archives-ouvertes.fr/hal-00936024/file/A-general-procedure-to-combine-estimators.pdf
"""
import numpy as np
from policies import expected_q_max
from scipy.stats import pearsonr


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


def estimate_variances_and_correlations(delta_mf, q_fn, env, gamma, X, Sp1, transition_model_fitter,
                                        alpha_mf_prev, number_of_bootstrap_samples=10):
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



