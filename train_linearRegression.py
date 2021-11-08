###Importing the libraries for the preprocessing
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def linReg(data,val_size,test_size, save_model=False):
    df_new = data.copy()
    ###Splitting the set into dependent and independent set
    X = df_new.drop("price",axis=1)
    Y = df_new.loc[:,"price"]
    ###Creating the train split and test split
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=test_size, random_state=42)
    ###Creating Validation split
    x_train_t, x_val, y_train_t, y_val = train_test_split(x_train, y_train, test_size=val_size,random_state=42)

    linreg = LinearRegression()
    linreg.fit(x_train_t,y_train_t)
    y_pred = linreg.predict(x_val)      ##Validation
    y_pred1 = linreg.predict(x_test)    ##Test
    
    ###the metrics
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
    metrics_linreg = pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])
    
    ###Saving the model
    if save_model is True:
        filepath = r"C:\Users\PSALISHOL\Documents\Ml projects\Car Price\Trained model\LinearRegressor"
        with open(filepath, "wb") as f:
            pickle.dump(object,f)
    print("LinearRegression")
    return metrics_linreg
        
###Training the model
if __name__ == '__main__':
    new_d = pd.read_csv(r"C:\Users\PSALISHOL\Documents\Ml projects\Car Price\input\Car Price(Preprocessed).csv")
    linReg(new_d,test_size=0.3,val_size=0.3, save_model=False)