#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Necessary packages
import numpy as np
import math
import pandas as pd
from utils import binary_sampler
from keras.datasets import mnist


def data_loader (data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  file_name = 'data/all_data/running_code_on/'+ data_name +'.csv'
  data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)

  no, dim = data_x.shape
  testing_size = 0.20

  # Introduce missing data
  testing_row_size = int (no * (testing_size))
  training_row_size = no - testing_row_size
  
  miss_rate = 0.5
  data_m = binary_sampler(1 - miss_rate, testing_row_size, dim-1)
    
  data_m = np.concatenate((data_m, np.ones((testing_row_size,1))), 1)
  data_all_ones = np.ones((training_row_size,dim))
  overall_data = np.vstack((data_all_ones,data_m))
  miss_data_x = data_x.copy()
  miss_data_x[overall_data == 0] = np.nan
  return data_x, miss_data_x, overall_data

