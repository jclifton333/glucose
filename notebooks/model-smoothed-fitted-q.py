
# coding: utf-8

# In[1]:


import sys
#sys.path.append('/mnt/c/Users/Jesse/Desktop/glucose')
#sys.path.append('/mnt/c/Users/Jesse/Desktop/glucose/src')
#sys.path.append('/mnt/c/Users/Jesse/Desktop/glucose/src/policies')
sys.path.append('/users/jclih/glucose')
sys.path.append('/users/jclih/glucose/src')
sys.path.append('/users/jclih/glucose/policies')


# In[2]:


from policies.policies import model_smoothed_fitted_q
from Glucose import Glucose
from transition_estimation import MultivariateLinear
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel
import numpy as np


# In[ ]:


env = Glucose(horizon=50)
transition_model_fitter = MultivariateLinear
gamma = 0.9
number_of_value_iterations = 1


# In[ ]:


done = False
env.reset()
pairwise_kernels_ = pairwise_kernels(np.vstack(env.X), metric='rbf')
kernel_sums = np.sum(pairwise_kernels_, axis=0)
pairwise_kernels = np.multiply(pairwise_kernels_, 1 / kernel_sums)
while not done:
    action, pairwise_kernel_, kernel_sums =         model_smoothed_fitted_q(env, gamma, RandomForestRegressor, number_of_value_iterations, transition_model_fitter,
                                pairwise_kernels_, kernel_sums)
    _, r, done = env.step(action)

