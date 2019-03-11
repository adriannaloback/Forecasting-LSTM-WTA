#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:20:01 2019

@author: adrianna
"""

import numpy as np
from sklearn.metrics import mean_squared_error

class EvalMetrics():
    """A class for for evaluating predictive model performance"""

    def __init__(self,y_true,y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_MAE(self):
        '''
        Returns MAE: Mean absolute error for validation set (in same units as input)
        '''
        return np.mean(np.abs(self.y_true - self.y_pred))

    def get_RMSE(self):
        '''
        Returns RMSE: Root mean squared error for validation set (in same units as input)
        '''
        return mean_squared_error(self.y_true, self.y_pred)
    
    def get_MAPE(self):
        '''
        Returns MAPE: Mean absolute percentage error (units: %)
        '''
        y_true, y_pred = check_arrays(self.y_true, self.y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
