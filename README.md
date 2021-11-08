# Volkwagen-Car-Price-Prediction
This project attempts to model and predict Car prices made by volkswagen. main aim of the project is to analyse the main feature in predicting the price of the car made by this company.
#### project status: Active
#### Current F1_score: 0.9445
#### RMSE: 0.234


## PROJECT PIPELINE
![Logo](https://github.com/psalishol/Volkwagen-Car-Price-Prediction/blob/main/Rendered%20Plots/project%20pipeline.PNG)

### Technologies used
- python: Scikit Learn, Matplotlib, Seaborn, pandas and Numpy


### Data and features
The data used for this project is gotten from kaggle.

**Data features:** Year, model, mileage, fuelType, mpg, transmission, tax, enginesize



### Exploration and Insight
- The data gotten has way too much outliers which really affected the model performance for the baseline model. A way was devised to remove and impute the outliers. The interquantile range was computed, and then values below the lower boundary and upper boundary is replaced with a NaN value after which was imputed using iterativeImputer using Randomforest as estimator, and that significantly improve the model.
- Year and mpg has the highest score in the prediction of the car price and fueltype the lowest
![Logo](https://github.com/psalishol/Volkwagen-Car-Price-Prediction/blob/main/Rendered%20Plots/Feature%20Importance.png)
- The company made the income in 2020, in which average price car was sold is 24,000 and the lowest in 2000
![Logo](https://github.com/psalishol/Volkwagen-Car-Price-Prediction/blob/main/Rendered%20Plots/Average%20Selling%20Price.png)
- California model has the highest price compared to others, it only uses diesel as Fuel which might be another reason for hike in the price. it is presumably seen to be one of the lowest purchased model by Volkswagen.
![Logo](https://github.com/psalishol/Volkwagen-Car-Price-Prediction/blob/main/Rendered%20Plots/Price%20and%20model.png)
- Golf on the other hand is the most purchased model, the price is relatively less to California model andd that made it easily purchasable by avaerage people.
![Logo](https://github.com/psalishol/Volkwagen-Car-Price-Prediction/blob/main/Rendered%20Plots/Purchase%20Count.png)
