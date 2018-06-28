# -*- encoding: utf-8 -*-
"""
==========
Rossman sales (Sequential)
==========
"""

import multiprocessing
import shutil

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import warnings
from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
import sklearn.metrics
import pandas as pd
import autosklearn.regression


tmp_folder = '/tmp/autosklearn_rossman_example_tmp_seq'
output_folder = '/tmp/autosklearn_rossman_example_out_seq'

def main():
    data_train = pd.read_csv("../Data/Rossman/train.csv",low_memory=False)
    data_test = pd.read_csv("../Data/Rossman/test.csv",low_memory=False)
    store = pd.read_csv("../Data/Rossman/store.csv", low_memory = False)
    closed_store_data = data_test["Id"][data_test["Open"] == 0].values
    data_train.StateHoliday = data_train.StateHoliday.replace(0,'0')
    data_test.StateHoliday = data_test.StateHoliday.replace(0,'0')
    data_train['Year'] = data_train['Date'].apply(lambda x: int(x[:4]))
    data_train['Month'] = data_train['Date'].apply(lambda x: int(x[5:7]))
    data_train["HolidayBin"] = data_train.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})
    store["Assortment"] = store.Assortment.map({"0": 0, "a": 1, "b": 2, "c": 3})
    store["StoreType"] = store.StoreType.map({"0": 0, "a": 1, "b": 2, "c": 3})
    del data_train['Date']
    del data_train['StateHoliday']
    data_test['Year'] = data_test['Date'].apply(lambda x: int(x[:4]))
    data_test['Month'] = data_test['Date'].apply(lambda x: int(x[5:7]))
    data_test["HolidayBin"] = data_test.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})
    del data_test['Date']
    del data_test['StateHoliday']
    X_train = data_train.copy()
    X_train.drop(["Sales", "Customers"], axis = 1, inplace = True)


    print (store.head(5))


    y_train = data_train['Sales'].copy()

    X_test = data_test.copy()
    X_test.drop(["Id"], axis = 1, inplace = True)



    new_train = pd.merge(X_train, store[["Store", "StoreType", "Assortment", "CompetitionDistance"]], on = 'Store', how = 'left')
    new_test = pd.merge(X_test, store[["Store", "StoreType", "Assortment", "CompetitionDistance"]], on = 'Store', how = 'left')





    # data_train_b = pd.read_csv('../Data/Rossman/train.csv', low_memory=False)
    # print (data_train_b.head())
    # store = pd.read_csv('../Data/Rossman/store.csv', low_memory=False)
    # data_test_b = pd.read_csv('../Data/Rossman/test.csv', low_memory=False)
    # store = store.drop(["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear","Promo2SinceYear", "PromoInterval"], axis = 1)
    # store.StoreType = store.StoreType.map({"0": 0, "a": 1, "b": 2, "c": 3, "d" :4})
    # store.Assortment = store.Assortment.map({"0": 0, "a": 1, "b": 2, "c": 3})
    # data_train = data_train_b.copy()
    # data_train['Year'] = data_train.Date.apply(lambda x: x.split('-'))
    # data_train['Month'] = data_train.Year.apply(lambda x: int(x[1]))
    # data_train['Day'] = data_train.Year.apply(lambda x: int(x[2]))
    # data_train['Year'] = data_train.Year.apply(lambda x: int(x[0]))
    # data_train = data_train.drop(["Date"], axis = 1)
    # data_train.StateHoliday = data_train.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1, "d":1})
    # new_train = pd.merge(data_train[['Store','Sales', 'DayOfWeek','Promo','StateHoliday','SchoolHoliday','Year', 'Month','Day']],store,on = 'Store',how ='left')
    #
    # data_test = data_test_b.copy()
    # data_test['Year'] = data_test.Date.apply(lambda x: x.split('-'))
    # data_test['Month'] = data_test.Year.apply(lambda x: int(x[1]))
    # data_test['Day'] = data_test.Year.apply(lambda x: int(x[2]))
    # data_test['Year'] = data_test.Year.apply(lambda x: int(x[0]))
    # data_test = data_test.drop(["Date"], axis = 1)
    # data_test.StateHoliday = data_test.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1, "d":1})
    # new_test = pd.merge(data_test[['Store', 'DayOfWeek','Promo','StateHoliday','SchoolHoliday','Year', 'Month','Day']],store,on = 'Store',how ='left')
    #
    # X_test = new_test
    # y_train = new_train["Sales"]
    # X_train = new_train.drop(columns = ["Sales"],axis =1)
    print (new_train.head())
    print (new_test.head())
    feature_types = (['numerical']) + (['categorical']*4) + (['numerical']*2) + (['categorical']*3) +(['numerical'])

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=3600,
        per_run_time_limit=360,
        ml_memory_limit = 3072,
        tmp_folder=tmp_folder,
        initial_configurations_via_metalearning = 0,
        output_folder=output_folder,
    )
    automl.fit(new_train, y_train, dataset_name='rossman',
               feat_type=feature_types)

    print(automl.show_models())

    predictions = automl.predict(new_test, n_jobs = 4)

    print('\nStatistics: \n', automl.sprint_statistics())

    df = pd.DataFrame({"Id": data_test.Id, "Sales": predictions})
    df.to_csv('predictions.csv', index = 0)

if __name__ == '__main__':
    main()
