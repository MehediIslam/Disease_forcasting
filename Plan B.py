#Importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.statespace.sarimax import SARIMAX 

#Load specific evaluation tools 
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 
  
#Read the Dengue cases dataset 
disease = pd.read_csv('C:/Users/Ruposh/Desktop/DengueCases.csv', 
                       index_col ='Time Period', 
                       parse_dates = True) 
  
#Print the first five rows of the dataset 
disease.head() 
  
#ETS Decomposition 
result = seasonal_decompose(disease['Cases'],   
                            model ='additive')
  
#ETS plot  
result.plot() 
plt.show()

==============================================

# Import the library 
from pmdarima import auto_arima 
  
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 

# Fit auto_arima function to AirPassengers dataset 
stepwise_fit = auto_arima(disease['Cases'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 
  
# To print the summary 
stepwise_fit.summary()   
#-------------------	Output	----------------------#



# Split data into train / test sets 
train = disease.iloc[:len(disease)-12] 
test = disease.iloc[len(disease)-12:] # set one year(12 months) for testing 
  
# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set 
 
model = SARIMAX(train['Cases'],  
                order = (0, 1, 1),  
                seasonal_order =(2, 1, 1, 12)) 
  
result = model.fit() 
result.summary()
#-------------------	Output	----------------------#
=============================================================

start = len(train) 
end = len(train) + len(test) - 1
  
# Predictions for one-year against the test set 
predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# plot predictions and actual values 
test['Cases'].plot(legend = True , label='Observed') 
predictions.plot(legend = True) 
plt.title('Evaluate Arima Model (Actual vs Prediction)')	
plt.ylabel('Number of Cases')
plt.show()


# Calculate root mean squared error 
rmse(test["Cases"], predictions) 
  
# Calculate mean squared error 
mean_squared_error(test["Cases"], predictions) 
===========================================================

# Train the model on the full dataset 
model = model = SARIMAX(disease['Cases'],  
                        order = (0, 1, 1),  
                        seasonal_order =(2, 1, 1, 12)) 
result = model.fit() 
  
# Forecast for the next 3 years 
forecast = result.predict(start = len(disease),  
                          end = (len(disease)-1) + 3 * 12,  
                          typ = 'levels').rename('Forecast') 
  
# Plot the forecast values 
disease['Cases'].plot(figsize = (12, 5), label='Observed', legend = True) 
forecast.plot(legend = True) 
plt.title('Probable Dengue Outbreaks')	
plt.ylabel('Number of Cases')			
plt.show()