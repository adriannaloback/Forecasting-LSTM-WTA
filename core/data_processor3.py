import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, vim_file, split, target_col):
        dataframe1      = pd.read_csv(filename, delim_whitespace=True)
        dataframe2      = pd.read_csv(vim_file, ',', header=None)
        i_split         = int(len(dataframe1) * split)
        data1_train     = dataframe1.get(target_col).values[:i_split]
        data2_train     = dataframe2.get(list(dataframe2)).values[:i_split]
        self.data_train = np.concatenate((data1_train,data2_train), axis=1)
        data1_test      = dataframe1.get(target_col).values[i_split:]
        data2_test      = dataframe2.get(list(dataframe2)).values[i_split:]
        self.data_test  = np.concatenate((data1_test,data2_test), axis=1)
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows (sliding windows)
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows     = np.array(data_windows).astype(float)
        data_windows_raw = data_windows #will use for de-normalising
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]     #input window sequence on which predictions will be generated
        y = data_windows[:, -1, [0]] #true last value in window sequence to compare model predictions to
        p0_vals = data_windows_raw[:, 0, [0]]
        return x,y,p0_vals

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows (sliding windows)
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]] #assumes 0th col is target var to predict
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
