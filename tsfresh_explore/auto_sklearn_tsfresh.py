"""
==========
Airlines (tsfresh + lags as exog)
==========
"""
import sklearn.metrics
import pandas as pd
import autosklearn.regression
import autosklearn.metrics
import numpy as np
import math
import time
from tqdm import tqdm
import warnings
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
from scipy.ndimage.interpolation import shift
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

tmp_folder = '/tmp/autosklearn_rossman_example_tmp_seq'
output_folder = '/tmp/autosklearn_rossman_example_out_seq'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# def plot_results(time_series,train_prediction, test_prediction, y_test):
#
#     """
#     Plots the original time series and it prediction
#     """
#
#     train_pr = train_prediction
#     test_pr = test_prediction
#     ts = time_series
#
#     plt.figure(figsize=(10,7))
#
#     plt.plot(ts, label = "true")
#
#     plt.plot(np.concatenate([train_pr, test_pr])[1:],
#          label = "predictions, \n MAPE = {0}, MAE = {1}".format(
#              np.round(mean_absolute_percentage_error(test_pr, y_test), 3),
#              np.round(mean_absolute_error(test_pr, y_test), 3)))
#
#
#     plt.axvline(x=119, label = "train-test-split", color = 'r')
#
#     plt.xlabel("time", size = 20)
#     plt.ylabel("value", size = 20)
#
#
#     plt.legend(fontsize = 15)
#
#     plt.show()
#     pass


def fit_rolling_auto_sklearn(y_train, max_timeshift = 10, rolling_direction = 1, params = None, my_dict_of_features=None):


    exog_lag = np.hstack((shift(np.concatenate([y_train,[0]]),shift = 1, cval = 0.0).reshape(-1,1),
               shift(np.concatenate([y_train,[0]]),shift = 2, cval = 0.0).reshape(-1,1),
               shift(np.concatenate([y_train,[0]]),shift = 3, cval = 0.0).reshape(-1,1),
               shift(np.concatenate([y_train,[0]]),shift = 12, cval = 0.0).reshape(-1,1)))

    df_shift, y = make_forecasting_frame(y_train, kind = "price",
                        max_timeshift = max_timeshift, rolling_direction = rolling_direction)
    X_train = extract_features(df_shift, column_id="id", column_sort="time",
                                    column_value="value", impute_function=impute, show_warnings=False,
                                  default_fc_parameters = my_dict_of_features,disable_progressbar = True)
    X_train.dropna(axis = 1, inplace = True)
    X_train = np.array(X_train)
    ts = y_train[2:]
    exog = np.hstack((X_train[:-1],exog_lag[2:-1]))
    # print (exog)
    last_exog = np.concatenate([X_train[-1],exog_lag[-1]]).reshape(1,-1)
    feature_types = (['numerical']*8) 
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=1200,
        per_run_time_limit=120,
        ml_memory_limit = 2048,
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        initial_configurations_via_metalearning = 0,
    )
    automl.fit(exog, ts, dataset_name='airlines', feat_type=feature_types,metric = autosklearn.metrics.mean_squared_error)

    predict_in_sample = automl.predict(exog)

    print(automl.show_models())
    print('\nStatistics: \n', automl.sprint_statistics())

    return automl, last_exog, predict_in_sample

def predict_rolling(model, last_exog, y_train, forecast_horizont, max_timeshift = 10, rolling_direction = 1, my_dict_of_features =None):

        """
        Predicting values on the next forecast_horizont values
        """
        predictions = np.empty(forecast_horizont)
        predictions[0] = model.predict(last_exog)
        for it in range(1,forecast_horizont):

            y_train = np.append(y_train, predictions[it-1])
            exog_lag = np.hstack((shift(np.concatenate([y_train,[0]]),shift = 1, cval = 0.0).reshape(-1,1),
                       shift(np.concatenate([y_train,[0]]),shift = 2, cval = 0.0).reshape(-1,1),
                       shift(np.concatenate([y_train,[0]]),shift = 3, cval = 0.0).reshape(-1,1),
                       shift(np.concatenate([y_train,[0]]),shift = 12, cval = 0.0).reshape(-1,1)))

            df_shift, y = make_forecasting_frame(y_train, kind = "price", max_timeshift = max_timeshift, rolling_direction = rolling_direction)

            X_train = extract_features(df_shift, default_fc_parameters = my_dict_of_features, column_id="id", column_sort="time", disable_progressbar = True, column_value="value", impute_function=impute, show_warnings=False)

            X_train.dropna(axis = 1, inplace = True)
            X_train = np.array(X_train)
            ts = y_train[2:]
            exog = np.concatenate([X_train[-1],exog_lag[-1]]).reshape(1,-1)

            y_pred = model.predict(exog)

        predictions[it] = y_pred

        return predictions

def main():
    df = pd.read_csv('AirPassengers.csv', usecols=[1])
    middle_point = 120
    data_train = np.array(df.iloc[:middle_point]).squeeze()
    data_test = np.array(df[middle_point:]).squeeze()
    data = np.concatenate([data_train, data_test])
    params = {}
    params['pyramid_mode'] = 'stepwise'
    params['ic'] = 'oob'
    params['period'] = 12
    params['random_state'] = 42
    params['n_fits'] = 10
    params['trend'] = 'c'
    params['n_jobs'] = 1
    params['scoring'] = 'mae'
    params['out_of_sample_size'] = 10
    params['dynamic'] = False
    my_dict_of_features = {'fft_coefficient':[{"coeff": 1, "attr": 'real'}], 'kurtosis':None,
                           'agg_linear_trend':[{"attr": 'pvalue', "chunk_len": 1, "f_agg": 'mean'},
                                               {"attr": 'pvalue', "chunk_len": 1, "f_agg": 'max'}]}

    model, last, predict_in_sample = fit_rolling_auto_sklearn(y_train=data_train, params=params, my_dict_of_features=my_dict_of_features)
    predictions = predict_rolling(model,last,data_train,24, my_dict_of_features=my_dict_of_features)
    print (predictions.shape)
    print ("Predictions:", predictions)

if __name__ == '__main__':
    main()
