"""
For estimating transition densities (model-based RL).

Refer to logistic function parameters as theta and multivariate normal mean parameters as beta.

m: number of mixture components
p: feature dimension
q: target dimension
"""


import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


def expit_linear_probability(theta_dot_x):
  return expit(theta_dot_x)


def linear_multivariate_normal_density(sigma, beta_dot_x, y):
  y_minus_mean = y - beta_dot_x
  term_in_exponent = -np.dot(y_minus_mean, y_minus_mean) / (2 * np.power(sigma, 2))
  normalizing_constant = 1 / np.power(sigma, len(x))
  return term_in_exponent * normalizing_constant


def log_likelihood_at_one_data_point(sigma_vec, theta_dot_x, beta_dot_x, y):
  """

  :param sigma_vec: m-length array of multivariate normal variances.
  :param theta_dot_x:
  :param beta_dot_x:
  :param y: q-length target array
  :return:
  """
  log_likelihood = 0.0
  m = len(sigma_vec)
  for mixture_component in range(m):
    sigma_m = sigma_vec[m]
    theta_dot_x_m = theta_dot_x[m, :]
    beta_dot_x_m = beta_dot_x[m, :]
    mixture_probability = expit_linear_probability(theta_dot_x_m)
    density_at_mixture_component = linear_multivariate_normal_density(sigma_m, beta_dot_x_m, y)
    log_likelihood += mixture_probability * density_at_mixture_component
  return log_likelihood


def log_likelihood(sigma_vec, theta, beta, X, Y):
  X_dot_theta = np.dot(X, theta)
  X_dot_beta = np.dot(X, beta)
  log_likelihood = 0.0
  for theta_dot_x, beta_dot_x, y in zip(X_dot_theta, X_dot_beta, Y):
    log_likelihood += log_likelihood_at_one_data_point(sigma_vec, theta_dot_x, beta_dot_x)
  return log_likelihood


def get_mdr_parameters_from_2d_array(arr, m, p, q):
    sigma_vec = arr[0, :]
    theta_array = arr[1:(p*q + 1), :].reshape((m, p, q))
    beta_array = arr[(p*q)+1:, :].reshape((m, p))
    return {'sigma_vec': sigma_vec, 'theta_array': theta_array, 'beta_array': beta_array}


def mixture_density_regression(X, Y, number_of_mixture_components):
  p = X.shape[1]
  q = Y.shape[1]

  def negative_log_likelihood(parameter):
    parameters = get_mdr_parameters_from_2d_array(parameter, number_of_mixture_components, p, q)
    sigma_vec = parameters['sigma_vec']
    theta_array = parameters['theta_array']
    beta_array = parameters['beta_array']
    return -1 * log_likelihood(sigma_vec, theta_array, beta_array, X, Y)

  x0 = np.zeros((1 + p*q + p, m))
  res = minimize(negative_log_likelihood, x0=x0, method='L-BFGS-B')
  return get_mdr_parameters_from_2d_array(res.x, number_of_mixture_components, p, q)


class MDR(object):

  def __init__(self, number_of_mixture_components=3)
    self.number_of_mixture_components = number_of_mixture_components
    self.model_is_fit = False
    self.sigma_vec = None
    self.theta_array = None
    self.beta_array = None
    self.output_dimension = None

  def fit(self, X, Y):
    res = mixture_density_regression(X, Y, self.number_of_mixture_components)
    self.sigma_vec = res['sigma_vec']
    self.theta_array = res['theta_array']
    self.beta_array = res['beta_array']
    self.output_dimension = Y.shape[1]
    self.model_is_fit = True

  def simulate_from_fitted_model(self, x):
    theta_dot_x = np.dot(self.theta_array, x)
    mixture_component_probabilities = [expit_linear_probability(theta_dot_x_m) for theta_dot_x_m in theta_dot_x]
    randomly_chosen_mixture_component = np.random.multinomial(1, pvals=mixture_component_probabilities)
    beta_at_mixture_component = self.beta_array[randomly_chosen_mixture_component, :]
    sigma_at_mixture_component = self.sigma_vec[randomly_chosen_mixture_component]
    y_simulated = np.random.multivariate_normal(np.dot(beta_at_mixture_component, x),
                                                sigma_at_mixture_component*np.eye(self.output_dimension))
    return y_simulated

  def roll_out_glucose_policy(self, initial_x, policy, rollout_depth):
    """
    Assume initial_x is a Glucose feature, and policy maps Glucose features to binary actions.  Using fitted model,
    roll out one random trajectory from initial_x under policy.

    :param initial_x:
    :param policy:
    :param rollout_depth:
    :return:
    """
    assert self.model_is_fit

    x = initial_x
    last_action = x[-2]
    last_state = x[:3]
    X = np.zeros((0, 9))  # For collecting observations
    for t in range(rollout_depth):
      next_state = self.simulate_from_fitted_model(x)
      next_and_last_state = np.concatenate((next_state, last_state, last_action))
      action = policy(next_and_last_state)
      x = np.concatenate((next_and_last_state, action))
      last_action = action
      X = np.vstack((X, x))
    return X




