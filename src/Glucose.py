import numpy as np


class Glucose(object):
  NUM_STATE = 8
  MAX_STATE = 1000 * np.ones(NUM_STATE)
  MIN_STATE = np.zeros(NUM_STATE)
  NUM_ACTION = 2

  # Generative model parameters
  MU_GLUCOSE = 250
  SIGMA_GLUCOSE = 25
  INS_PROB = 0.3
  MU_FOOD = 0
  SIGMA_FOOD = 10
  MU_ACTIVITY = 0
  SIGMA_ACTIVITY = 10
  MU_ACTIVITY_MOD = 31
  SIGMA_ACTIVITY_MOD = 5
  SIGMA_NOISE = 5
  COEF = np.array([10, 0.9, 0.1, 0.1, -0.01, -0.01, -10, -4])

  # Test states
  HYPOGLYCEMIC = np.array([50, 0, 33, 50, 0, 0, 0, 0])
  HYPERGLYCEMIC = np.array([200, 0, 30, 200, 0, 0, 78, 0])

  def __init__(self, horizon):
    self.R = []  # List of rewards at each time step
    self.A = []  # List of actions at each time step
    self.S = []  # List of raw states at each time step
    self.t = 0
    self.horizon = horizon

  @staticmethod
  def reward_function(s_prev, s):
    """

    :param sprev: state vector at previous time step
    :param s: state vector
    :return:
    """
    new_glucose = s[0]
    last_glucose = sprev[0]

    # Reward from this timestep
    r_1 = (new_glucose < 70) * (-0.005 * new_glucose**2 + 0.95 * new_glucose - 45) + \
         (new_glucose >= 70) * (-0.00017 * new_glucose**2 + 0.02167 * new_glucose - 0.5)

    # Reward from previous timestep
    r2 = (last_glucose < 70)*(-0.005*last_glucose**2 + 0.95*last_glucose - 45) + \
         (last_glucose >= 70)*(-0.00017*last_glucose**2 + 0.02167*last_glucose - 0.5)
    return r1 + r2

  def generate_food_and_activity(self):
    """

    :return:
    """
    food = np.random.normal(Glucose.MU_FOOD, Glucose.SIGMA_FOOD)
    food = np.multiply(np.random.random() < 0.2, food)
    activity = np.random.normal(Glucose.MU_ACTIVITY, self.SIGMA_ACTIVITY)
    activity = np.multiply(np.random.random() < 0.2, activity)
    return food, activity

  def reset(self):
    """

    :return:
    """
    food, activity = self.generate_food_and_activity()
    glucose = np.random.normal(Glucose.MU_GLUCOSE, Glucose.SIGMA_GLUCOSE)
    s3 = 100.0
    s4, s5, s6, s7 = 0.0, 0.0, 0.0, 0.0
    s = np.array([glucose, food, activity, s3, s4, s5, s6, s7])

    self.S.append(s)
    return

  def next_state_and_reward(self, action):
    """

    :param action:
    :return:
    """
    # Transition model depends on previous two actions and states
    last_state = self.S[-1]
    if self.t > 1:
      state_before_last = self.S[-2]
      last_action = self.A[-1]
    else:
      state_before_last = last_state
      last_action = 0

    x = np.hstack(([1], last_state[:2], state_before_last[1], last_state[2], state_before_last[2], action, last_action))
    glucose = np.dot(x, Glucose.COEF) + np.random.normal(0, Glucose.SIGMA_NOISE)
    food, activity = self.generate_food_and_activity()
    s3 = self.S[-1][0]
    s4 = s5 = s6 = self.S[-1][1]
    s7 = self.S[-1][2]

    s_prev = self.S[-1]
    s_next = np.array([glucose, food, activity, s3, s4, s5, s6, s7])
    reward = self.reward_function(s_prev, s_next)
    return s_next, reward

  def get_state_history_as_array(self):
    """
    Return past states as an array with blocks [ lag 1 states, states]
    :return:
    """
    return np.column_stack((np.vstack(self.S[:-2]), np.vstack(self.S[1:-1])))

  def step(self, action):
    self.t += 1
    done = self.t == self.horizon
    s_next, reward = self.next_state_and_reward(action)
    self.S.append(s_next)
    self.R.append(reward)
    self.A.append(action)
    return done


