# -*- encoding: utf-8 -*-
"""
==========
Rossman sales (Parallel)
==========
"""

import multiprocessing
import shutil

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
import sklearn.metrics
import pandas as pd
import autosklearn.regression


tmp_folder = '/tmp/autosklearn_rossman_example_tmp_parallel'
output_folder = '/tmp/autosklearn_rossman_example_out_parallel'


for dir in [tmp_folder, output_folder]:
    try:
        shutil.rmtree(dir)
    except OSError as e:
        pass

def get_spawn_regressor(X_train, y_train, feature_types):
    def spawn_regressor(seed, dataset_name):
        """Spawn a subprocess.

        auto-sklearn does not take care of spawning worker processes. This
        function, which is called several times in the main block is a new
        process which runs one instance of auto-sklearn.
        """

        # Use the initial configurations from meta-learning only in one out of
        # the four processes spawned. This prevents auto-sklearn from evaluating
        # the same configurations in four processes.
        if seed == 0:
            initial_configurations_via_metalearning = 0
            smac_scenario_args = {}
        else:
            initial_configurations_via_metalearning = 0
            smac_scenario_args = {'initial_incumbent': 'RANDOM'}

        # Arguments which are different to other runs of auto-sklearn:
        # 1. all classifiers write to the same output directory
        # 2. shared_mode is set to True, this enables sharing of data between
        # models.
        # 3. all instances of the AutoSklearnClassifier must have a different seed!
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=4800,
            per_run_time_limit=480,
            ml_memory_limit = 2048,
            shared_mode=True, # tmp folder will be shared between seeds
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            delete_tmp_folder_after_terminate=False,
            ensemble_size=0, # ensembles will be built when all optimization runs are finished
            initial_configurations_via_metalearning = 0,
            seed=seed,
            smac_scenario_args=smac_scenario_args,
        )
        automl.fit(X_train, y_train, metric = autosklearn.metrics.mean_squared_error, feat_type=feature_types, dataset_name='rossman')
    return spawn_regressor


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
    feature_types = (['numerical']) + (['categorical']*4) + (['numerical']*2) + (['categorical']*3) +(['numerical'])

    processes = []
    spawn_regressor = get_spawn_regressor(new_train, y_train,feature_types)
    for i in range(2): # set this at roughly half of your cores
        p = multiprocessing.Process(target=spawn_regressor, args=(i, 'rossman'))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print('Starting to build an ensemble!')
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=480,
        per_run_time_limit=60,
        ml_memory_limit = 2048,
        shared_mode=True,
        ensemble_size = 50,
        ensemble_nbest=200,
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        initial_configurations_via_metalearning = 0,
        seed=1,
    )


    # Both the ensemble_size and ensemble_nbest parameters can be changed now if
    # necessary
    automl.fit_ensemble(
        y_train,
        task=REGRESSION,
        metric = autosklearn.metrics.mean_squared_error,
        precision='32',
        dataset_name='rossman',
        ensemble_size=20,
        ensemble_nbest=50,
    )

    print(automl.show_models())

    predictions = automl.predict(new_test, n_jobs = 4)
    df = pd.DataFrame({"Id": data_test.Id, "Sales": predictions})
    df.to_csv('predictions.csv', index = 0)

if __name__ == '__main__':
    main()
