# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:47:44 2023

@author: Admin
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting


# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# read data (make sure .csv in folder)
df = pd.read_csv('/Users/Admin/OneDrive/Documents/social_network_ads_ex_csv')
#print(df.head(),'\n') # print first 5 rows of data