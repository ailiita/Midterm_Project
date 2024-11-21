# Midterm Project

## Data Description
The dataset contains hourly air pollutant data from 12 air-quality monitoring site. The data, provided by the Beijing Municipal Environmental Monitoring Center.  
This work is based on the datset from the Wanshouxigong station. 
The datasets can be downloaded [here](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data).    
It contains weather data (rain, pressure, temperatures, etc.) and six polluants concentrations (pm2.5, pm10, so2, no2, co, o3) measured from March 1, 2013, to February 28, 2017. 
- No: row number 
- year: year of data in this row 
- month: month of data in this row 
- day: day of data in this row 
- hour: hour of data in this row 
- PM2.5: PM2.5 concentration (ug/m^3)
- PM10: PM10 concentration (ug/m^3)
- SO2: SO2 concentration (ug/m^3)
- NO2: NO2 concentration (ug/m^3)
- CO: CO concentration (ug/m^3)
- O3: O3 concentration (ug/m^3)
- TEMP: temperature (degree Celsius) 
- PRES: pressure (hPa)
- DEWP: dew point temperature (degree Celsius)
- RAIN: precipitation (mm)
- wd: wind direction
- WSPM: wind speed (m/s)
- station: name of the air-quality monitoring site

## Context and Objective
The Air Quality Index (AQI) is a numerical scale used to communicate the quality of air and its potential impact on human health. It measures the concentration of specific air pollutants in the atmosphere. The values range from 0 to 500. The higher the AQI, the worse the air quality and the greater the potential health risks.
The AQI is divided into six categories that correspond to different levels of health concern:
- 0-50: **Good** – Air quality is considered satisfactory, little or no risk for health.
- 51-100: **Moderate** – Air quality is acceptable. Some pollutants may pose a moderate health concern for a very small number of people.
- 101-150: **Unhealthy for sensitive groups** – People with respiratory or heart conditions may begin to experience health effects.
- 151-200: **Unhealthy** – Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects.
- 201-300: **Very unhealthy** – Health alert: everyone may experience more serious health effects.
- 301-500: **Hazardous** – Health warning of emergency conditions; the entire population is likely to be affected.

The objective of this analysis is to predict air quality based on the available data. To achieve this, the AQI was calculated from the pollutant concentrations using the standard tables for each pollutant. The resulting numerical AQI values were then converted into class labels.  
Multiple analyses were conducted to predict the AQI, including the following approaches:
- Using all available features *(not shown in notebook)*
- Using only weather-related features *(not shown in notebook)*
- Using a combination of weather and pollutant features

 **Using all available features** : All models, including the simpler ones, demonstrated exceptionally high accuracy, approaching perfection (very strong overfitting even when tuning parameters). This suggests that the use case may not be particularly challenging or insightful.  
**Using only weather-related features** : All models, including Decision Tree, Random Forest, and XGBoost, exhibited low accuracy, ranging from 30% to 45%. These results suggest that the weather dataset may not be strongly predictive of the target variable.  
**Using a combination of weather and pollutant features** : Using a combination of weather and pollutant features enabled the development of a more meaningful use case. This approach involved tuning the models to optimize accuracy while addressing and reducing overfitting, resulting in a more robust and reliable predictive model.

## Model Selection and Parameter Tuning
I used two models to address this problem :
- Logistic Regression (tuned regularization parameter C)
- Random Forest (tuned `n_estimators`, `max_depth` and `min_samples_leaf`)

The model was trained using 60% of the dataset and 20% was used for validation. Random Forest demonstrated the highest accuracy and was selected to train the final model.

## Training the final model 
Model used for final training is a random forest with parameters : 
- `n_estimators` = 3
-  `max_depth` = 6
-  `rancom_state` = 1

The ***train.py*** file trains this model (80% for training, 20% for testing), it also performs cross-validation using 8 folds. Results :   
```
Cross-validation results :     
 0.858 +- 0.070    
Training final model    
 Validation accuracy : 0.926
```
The accuracy on fold 1 (0.6997) was much lower compared to the other folds, which have accuracies between 0.857 and 0.903.   
This suggests that the model might performs differently according to the subset of data like for fold 1 but overall it performed well on al other folds, getting closer to the validation accuracy.  
This difference can be explained by the class imbalance of our aqi data. 1st fold could've ended up with more examples of the minority class (*Good* and *Hazardous* for example), making it more difficult for the model to predict accurately, explaining why it got such a low accuracy.

## Web service
The model is deployed using flask.
To run the model locally execute:
```
pip install pipenv
pipenv install -r requirements.txt
pipenv run waitress-serve --listen=0.0.0.0:9696 predict:app
``` 
In another shell, execute : 
```
python prediction_test.py
```
***requirements.txt*** lists all the dependencies with their versions required to run the project. 
The ***prediction_test.py*** provides the results of a test using some data as input. ***prediction_test.ipynb*** is the notebook that displays the result. 

## Containerization
***Dockerfile***; ***Pipfile***, ***Pifile.lock*** are provided.
To build and start the service's Docker container, follow these instructions :
- Download Docker Desktop
- Execute :
```
pipenv install requirements.txt
docker built -t air-quality .
docker run -it -p 9696:9696 air-quality
```
In another shell, execute : 
```
python prediction_test.py
```















