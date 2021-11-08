###Importing the libraries for the preprocessing
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
def Randomforestreg(data, plot_fea_imp=False,save_model=False, verbose_=None,n_job_=None):

    r2_scor_mean = None
    dk_data = data.copy()
    ###Splitting the set into dependent and independent set
    X = dk_data.drop("price",axis=1)
    Y = dk_data.loc[:,"price"]
    ###Creating the train split and test split
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
    ###Creating Validation split
    x_train_t, x_val, y_train_t, y_val = train_test_split(x_train, y_train, test_size=0.3,random_state=42)

    
    """Tuning the Hyperparameters"""
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, 
                                    param_distributions = random_grid, 
                                    n_iter = 15,
                                    cv = 5, 
                                    verbose=verbose_, 
                                    random_state=42,
                                    n_jobs = n_job_
                ).fit(x_train_t,y_train_t)
        
    estimator = rf_random.best_estimator_
    estimator.fit(x_train_t,y_train_t)
    y_pred = estimator.predict(x_val)
    y_pred1 = estimator.predict(x_test)
        
    ###Creating Evaluation metrics
    metrics = {
        "Accuracy Test" : np.round(100 - mean_absolute_percentage_error(y_test,y_pred1),3),
        "Accuracy Validation" :np.round(100 - mean_absolute_percentage_error(y_val,y_pred),3),
        "R2_score Test" : r2_score(y_test,y_pred1),
        "R2_score Validation": r2_score(y_val,y_pred),
        "RMSE Test": np.sqrt(mean_squared_error(y_test,y_pred1)),
        "RMSE Validation": np.sqrt(mean_squared_error(y_val,y_pred)),
        "MAE Test": mean_absolute_error(y_test,y_pred1),
        "MAE Validation": mean_absolute_error(y_val,y_pred)
            }
    metrics_randf  = pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])
    ###Saving the model
    if save_model:
        filepath = r"C:\Users\PSALISHOL\Documents\Ml projects\Car Price\Trained model\RandomForestRegressor"
        with open(filepath, "wb") as f:
            pickle.dump(object,f)
        
    return metrics_randf
    
   
###Training the model
if __name__ == '__main__':
    new_d = pd.read_csv(r"C:\Users\PSALISHOL\Documents\Ml projects\Car Price\input\Car Price(Preprocessed).csv")
    Randomforestreg(new_d,save_model=True,verbose_=1,n_job_=-1)