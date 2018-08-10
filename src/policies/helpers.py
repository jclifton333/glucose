import numpy as np
import pdb


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
  for draw in range(num_draws):
    _, Xp1 = transition_model.simulate_from_fit_model_at_block(X)
    q_max_at_draw, _ = maximize_q_function_at_block(q_fn, Xp1, env)
    expected_q_max_ += (q_max_at_draw - expected_q_max_) / (draw + 1)
  return expected_q_max_


def add_crust_to_square_matrix(M):
  m = M.shape[0]
  M = np.vstack((M, np.zeros(m)))
  M = np.column_stack((M, np.zeros(m + 1)))
  return M


def update_pairwise_kernels_(pairwise_kernels_, kernel, kernel_sums, X):
  """

  :param pairwise_kernels_:
  :param kernel:
  :param kernel_sums: Normalizing constants (i.e. sum(pairwise_kernels, axis=0) before pairwise_kernels_ was divided
                      by same)
  :param X: array of vectors where last row is the new observation for which to compute pairwise kernels
  :return:
  """
  if X.shape[0] < 2:
    pairwise_kernels_ = np.array([[1.0]])
    kernel_sums = np.array([1.0])
  else:
    # Un-normalize
    pairwise_kernels_ = np.multiply(pairwise_kernels_, kernel_sums)

    # Get kernels at new state
    new_row = np.array([kernel(X[-1, :].reshape(1, -1), X[i, :].reshape(1, -1))[0] for i in range(X.shape[0])])
    pairwise_kernels_ = add_crust_to_square_matrix(pairwise_kernels_)
    pdb.set_trace()
    pairwise_kernels_[-1, :] = new_row
    pairwise_kernels_[:, -1] = new_row

    # Renormalize
    kernel_sums = np.concatenate((kernel_sums + new_row[:-1], [np.sum(new_row)]))
    pairwise_kernels_ = np.multiply(pairwise_kernels_, 1 / kernel_sums)

  return pairwise_kernels_, kernel_sums
