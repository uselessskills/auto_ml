# -*- encoding: utf-8 -*-
"""
==========
Rossman sales (Sequential)
==========
"""
import sklearn.metrics
import pandas as pd
import autosklearn.regression
import autosklearn.metrics

tmp_folder = '/tmp/autosklearn_rossman_example_tmp_seq'
output_folder = '/tmp/autosklearn_rossman_example_out_seq'

def main():
    data_train = pd.read_csv("../Data/Rossman/train.csv",low_memory=False)
    data_test = pd.read_csv("../Data/Rossman/test.csv",low_memory=False)
    store = pd.read_csv("../Data/Rossman/store.csv", low_memory = False)
    closed_store_data = data_test["Id"][data_test["Open"] == 0].values
    data_train.StateHoliday = data_train.StateHoliday.replace(0,'0')
    data_test.StateHoliday = data_test.StateHoliday.replace(0,'0')
    data_train = data_train.sample(70000)
    #print (data_train.tail())
    data_train['Year'] = data_train['Date'].apply(lambda x: int(x[:4]))
    data_train['Month'] = data_train['Date'].apply(lambda x: int(x[5:7]))
    data_train["HolidayBin"] = data_train.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})
    store["Assortment"] = store.Assortment.map({"0": 0, "a": 1, "b": 2, "c": 3})
    store["StoreType"] = store.StoreType.map({"0": 0, "a": 1, "b": 2, "c": 3, "d":4})
    del data_train['Date']
    del data_train['StateHoliday']
    data_test['Year'] = data_test['Date'].apply(lambda x: int(x[:4]))
    data_test['Month'] = data_test['Date'].apply(lambda x: int(x[5:7]))
    data_test["HolidayBin"] = data_test.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})
    data_test.Open = data_test.Open.replace(0,'1.0')
    del data_test['Date']
    del data_test['StateHoliday']
    X_train = data_train.copy()
    X_train.drop(["Sales", "Customers"], axis = 1, inplace = True)
    y_train = data_train['Sales'].copy()
    X_test = data_test.copy()
    X_test.drop(["Id"], axis = 1, inplace = True)
    new_train = pd.merge(X_train, store[["Store", "StoreType", "Assortment", "CompetitionDistance"]], on = 'Store', how = 'left')
    new_test = pd.merge(X_test, store[["Store", "StoreType", "Assortment", "CompetitionDistance"]], on = 'Store', how = 'left')
    print (new_train.head())
    print (new_test.head())

    #Start auto-sklearn
    feature_types = (['numerical']) + (['categorical']*4) + (['numerical']*2) + (['categorical']*3) +(['numerical'])
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=1200,
        per_run_time_limit=120,
        ml_memory_limit = 2048,
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        initial_configurations_via_metalearning = 0,
    )
    automl.fit(new_train, y_train, dataset_name='rossman', feat_type=feature_types)#,metric = autosklearn.metrics.mean_squared_error)

    print(automl.show_models())
    print('\nStatistics: \n', automl.sprint_statistics())

    predictions = automl.predict(new_test, n_jobs = 4)

    df = pd.DataFrame({"Id": data_test.Id, "Sales": predictions})
    df.to_csv('predictions_16_58.csv', index = 0)

if __name__ == '__main__':
    main()
