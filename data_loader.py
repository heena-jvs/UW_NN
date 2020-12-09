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
  file_name = 'data/'+data_name+'.csv'
  data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)

  # data_x = np.delete(data_x, [4,5], 1)


  # Parameters
  no, dim = data_x.shape

  """
  train_ind = math.floor(no*0.7)
  valid_ind = math.floor(no*0.8)

  train_data = data_x[0:train_ind,:]
  valid_data = data_x[(train_ind+1):valid_ind,:]
  test_data = data_x[(valid_ind+1):,:]

  # training dataset
  train_no, train_dim = train_data.shape
  train_m = binary_sampler(1-miss_rate, train_no, train_dim)
  miss_train = train_data.copy()
  miss_train[train_m == 0] = np.nan

  # validation dataset
  valid_no, valid_dim = valid_data.shape
  valid_m = binary_sampler(1 - miss_rate, valid_no, valid_dim)
  miss_valid = valid_data.copy()
  miss_valid[valid_m == 0] = np.nan

  # testing dataset
  test_no, test_dim = test_data.shape
  test_m = binary_sampler(1 - miss_rate, test_no, test_dim)
  miss_test = test_data.copy()
  miss_test[test_m == 0] = np.nan

  
  # Introduce missing data
  #data_m = binary_sampler(1-miss_rate, no, dim)
  #miss_data_x = data_x.copy()
  #miss_data_x[data_m == 0] = np.nan
      
  return train_data, valid_data, test_data, miss_train, miss_valid, miss_test, train_m, valid_m, test_m
  """

  # Introduce missing data
  data_m = binary_sampler(1 - miss_rate, no, dim-4)
  data_m = np.concatenate((data_m, np.ones((no,4))), 1)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m