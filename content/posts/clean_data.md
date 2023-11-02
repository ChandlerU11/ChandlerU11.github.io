+++
title = "Cleaning Up Years of Running Data"
date = "2023-08-19"
author = "Chandler Underwood"
description = "In this project I use several techniques to clean up and fill-in my running data from college that spans 6 years."
+++
## Introduction
For those of you that don't know me personally, I managed to run collegiately for 6 years (thanks Covid?). Over that time I logged a lot of miles and a lot Garmin activities. I still run quite a bit, but my days of trying to really burn the barn down are behind me. I'd like to build a dashboard to get some insights to my running trends during that time as a sort of "last hoorah", but sadly, a lot of my running data is missing and messy. I think cleaning it up will make for a great data science project to test my skills! Follow along here as I clean-up and fill-in my running data using various techniques such as pulling in outside data sources and training some ML models to predict missing values. 

## Data Read-in and Initial Exploration
Below is a first look at my running data from college. Over the span of 6 years, I went for a run at least 1,975 times! 

```python
import pandas as pd

data_path = "Activities 20"
df = pd.read_csv(data_path + "17.csv")
for i in range (18,23):
     df = pd.concat([df, pd.read_csv(data_path + str(i) + ".csv")])
df = df.reset_index().drop(columns = 'index')
df
```
{{< rawhtml >}}
<div  style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Activity Type</th>
      <th>Date</th>
      <th>Favorite</th>
      <th>Title</th>
      <th>Distance</th>
      <th>Calories</th>
      <th>Time</th>
      <th>Avg HR</th>
      <th>Max HR</th>
      <th>Avg Run Cadence</th>
      <th>...</th>
      <th>Min Temp</th>
      <th>Surface Interval</th>
      <th>Decompression</th>
      <th>Best Lap Time</th>
      <th>Number of Laps</th>
      <th>Max Temp</th>
      <th>Moving Time</th>
      <th>Elapsed Time</th>
      <th>Min Elevation</th>
      <th>Max Elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Running</td>
      <td>2017-12-31 10:48:59</td>
      <td>False</td>
      <td>Sevierville Running</td>
      <td>13.62</td>
      <td>1,462</td>
      <td>01:29:31</td>
      <td>0</td>
      <td>0</td>
      <td>175</td>
      <td>...</td>
      <td>0.0</td>
      <td>0:00</td>
      <td>No</td>
      <td>00:00.00</td>
      <td>14</td>
      <td>0.0</td>
      <td>01:29:30</td>
      <td>01:32:18</td>
      <td>958</td>
      <td>1,181</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Running</td>
      <td>2017-12-30 08:44:33</td>
      <td>False</td>
      <td>Sevierville Running</td>
      <td>5.83</td>
      <td>631</td>
      <td>00:41:23</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>...</td>
      <td>0.0</td>
      <td>0:00</td>
      <td>No</td>
      <td>00:00.00</td>
      <td>6</td>
      <td>0.0</td>
      <td>00:41:25</td>
      <td>00:41:35</td>
      <td>1,001</td>
      <td>1,174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Running</td>
      <td>2017-12-29 11:21:54</td>
      <td>False</td>
      <td>Sevierville Running</td>
      <td>8.22</td>
      <td>881</td>
      <td>00:50:45</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>...</td>
      <td>0.0</td>
      <td>0:00</td>
      <td>No</td>
      <td>00:00.00</td>
      <td>10</td>
      <td>0.0</td>
      <td>00:50:45</td>
      <td>00:51:27</td>
      <td>968</td>
      <td>1,167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Running</td>
      <td>2017-12-29 11:06:10</td>
      <td>False</td>
      <td>Sevierville Running</td>
      <td>1.97</td>
      <td>209</td>
      <td>00:13:42</td>
      <td>0</td>
      <td>0</td>
      <td>174</td>
      <td>...</td>
      <td>0.0</td>
      <td>0:00</td>
      <td>No</td>
      <td>00:00.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>00:13:41</td>
      <td>00:14:01</td>
      <td>1,028</td>
      <td>1,178</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Running</td>
      <td>2017-12-28 06:24:02</td>
      <td>False</td>
      <td>Moss Point Running</td>
      <td>7.37</td>
      <td>797</td>
      <td>00:52:04</td>
      <td>0</td>
      <td>0</td>
      <td>173</td>
      <td>...</td>
      <td>0.0</td>
      <td>0:00</td>
      <td>No</td>
      <td>00:00.00</td>
      <td>8</td>
      <td>0.0</td>
      <td>00:52:03</td>
      <td>00:52:16</td>
      <td>64</td>
      <td>135</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>Running</td>
      <td>2022-01-05 08:38:41</td>
      <td>False</td>
      <td>Starkville Running</td>
      <td>11.50</td>
      <td>1,014</td>
      <td>01:19:58</td>
      <td>147</td>
      <td>163</td>
      <td>178</td>
      <td>...</td>
      <td>57.2</td>
      <td>0:00</td>
      <td>No</td>
      <td>01:19:57.54</td>
      <td>1</td>
      <td>75.2</td>
      <td>01:19:48</td>
      <td>01:22:26</td>
      <td>223</td>
      <td>402</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>Running</td>
      <td>2022-01-04 10:30:42</td>
      <td>False</td>
      <td>Noxubee County Running</td>
      <td>11.01</td>
      <td>851</td>
      <td>01:07:48</td>
      <td>145</td>
      <td>169</td>
      <td>183</td>
      <td>...</td>
      <td>51.8</td>
      <td>0:00</td>
      <td>No</td>
      <td>02:01.19</td>
      <td>14</td>
      <td>69.8</td>
      <td>01:07:39</td>
      <td>01:23:25</td>
      <td>130</td>
      <td>201</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>Running</td>
      <td>2022-01-03 10:29:42</td>
      <td>False</td>
      <td>Starkville Running</td>
      <td>15.01</td>
      <td>1,349</td>
      <td>01:43:07</td>
      <td>148</td>
      <td>171</td>
      <td>179</td>
      <td>...</td>
      <td>37.4</td>
      <td>0:00</td>
      <td>No</td>
      <td>01:43:06.81</td>
      <td>1</td>
      <td>77.0</td>
      <td>01:43:04</td>
      <td>01:51:40</td>
      <td>89</td>
      <td>243</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>Running</td>
      <td>2022-01-02 09:07:55</td>
      <td>False</td>
      <td>Leon County Running</td>
      <td>8.01</td>
      <td>688</td>
      <td>00:56:25</td>
      <td>141</td>
      <td>156</td>
      <td>177</td>
      <td>...</td>
      <td>77.0</td>
      <td>0:00</td>
      <td>No</td>
      <td>56:24.51</td>
      <td>1</td>
      <td>86.0</td>
      <td>00:56:22</td>
      <td>00:58:52</td>
      <td>-94</td>
      <td>167</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>Running</td>
      <td>2022-01-01 09:45:26</td>
      <td>False</td>
      <td>Tallahassee Running</td>
      <td>7.00</td>
      <td>593</td>
      <td>00:51:11</td>
      <td>136</td>
      <td>159</td>
      <td>175</td>
      <td>...</td>
      <td>75.2</td>
      <td>0:00</td>
      <td>No</td>
      <td>51:10.60</td>
      <td>1</td>
      <td>84.2</td>
      <td>00:51:06</td>
      <td>00:56:12</td>
      <td>-59</td>
      <td>205</td>
    </tr>
  </tbody>
</table>
<p>1975 rows × 38 columns</p>
</div>
{{< /rawhtml >}}


Let's have a look at the columns in our dataset.

```python
df.columns
```
    Index(['Activity Type', 'Date', 'Favorite', 'Title', 'Distance', 'Calories',
           'Time', 'Avg HR', 'Max HR', 'Avg Run Cadence', 'Max Run Cadence',
           'Avg Pace', 'Best Pace', 'Total Ascent', 'Total Descent',
           'Avg Stride Length', 'Avg Vertical Ratio', 'Avg Vertical Oscillation',
           'Avg Ground Contact Time', 'Training Stress Score®', 'Avg Power',
           'Max Power', 'Grit', 'Flow', 'Avg. Swolf', 'Avg Stroke Rate',
           'Total Reps', 'Dive Time', 'Min Temp', 'Surface Interval',
           'Decompression', 'Best Lap Time', 'Number of Laps', 'Max Temp',
           'Moving Time', 'Elapsed Time', 'Min Elevation', 'Max Elevation'],
          dtype='object')



As you may have guessed, after reading Garmin's documentation, many of the data's attributes are not useful to us as they are not metrics taken for runs such as *Max Power* (a cycling metric) and *Avg Stroke Rate* (a swimming metric). In the cleanup and feature engineering section, we'll drop those and many others that aren't helpful for understanding my running performances. 

This first issue with this dataset is that the *Avg HR* and *Max HR* columns are populated with some zeros (see table), and I assure you that my heart was beating faster than that! The *Max Temp* and *Min Temp* columns also contain some zeroes. This is because I didn't have a fancy watch in the beginning of college that logged those metrics. Because of this, we can assume that the 0's populating those four columns are actually NULL values. Many of our columns should contain numerical data but instead contain strings such as *Min Elevation*, so we are going to fix that in the next section too. 

## Data Cleaning and Feature Engineering

First thing we need to do is drop those extrenious columns and correct the datatypes for our remaining columns. 


```python
import numpy as np

#Drop extranious columns
cols_to_keep = ['Date', 'Title', 'Time', 'Avg Pace', 'Best Pace', 'Distance',
       'Calories', 'Avg HR', 'Max HR', 'Avg Run Cadence',
       'Max Run Cadence', 'Total Ascent',
       'Total Descent', 'Avg Stride Length', 'Min Temp', 'Max Temp', 'Min Elevation',
       'Max Elevation']

df = df[cols_to_keep]

#Replace missing values with NaN for easy pandas manipulation
df = df.replace('--', np.nan)  #String Garmin uses in place of NaN
df = df.replace(0.0, np.nan)
df = df.replace(0, np.nan)

#Remove commas so we can convert these columns to numerical data
cols_to_clean = ['Calories', 'Total Ascent', 'Total Descent', 'Min Elevation', 'Max Elevation']
df[cols_to_clean] = df[cols_to_clean].replace({',':''}, regex=True)

#Conversion of columns to floats for use in models
def float_convert(col):
    df[col] = df[col].astype(float)

float_convert(cols_to_keep[5:])
```

There are a few important columns that are written in a time format that is useful for humans but not machines. Let's engineer some new features using them. 


```python
#Drop rows we don't have pacing data for
df = df[df['Avg Pace'].notna()]
df = df[df['Best Pace'].notna()]

#Convert values to float representing an equal amount of time in minutes
df['Total Run Time'] = [60 * float(x.split(':')[0]) + float(x.split(':')[1]) + (float(x.split(':')[2].split('.')[0])/60) for x in df['Time']]
df.drop(columns = 'Time', inplace = True)
df['Avg Pace'] = [float(x.split(':')[0]) + float(x.split(':')[1]) / 60 for x in df['Avg Pace']]
df['Best Pace'] = [float(x.split(':')[0]) + float(x.split(':')[1]) / 60 for x in df['Best Pace']]

#My college running days ended on the date below
df['Date'] = pd.to_datetime(pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d'))
df = df[df['Date'] < np.datetime64("2022-05-15")]
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1883 entries, 0 to 1974
    Data columns (total 18 columns):
     #   Column             Non-Null Count  Dtype         
    ---  ------             --------------  -----         
     0   Date               1883 non-null   datetime64[ns]
     1   Title              1883 non-null   object        
     2   Avg Pace           1883 non-null   float64       
     3   Best Pace          1883 non-null   float64       
     4   Distance           1883 non-null   float64       
     5   Calories           1883 non-null   float64       
     6   Avg HR             607 non-null    float64       
     7   Max HR             607 non-null    float64       
     8   Avg Run Cadence    1883 non-null   float64       
     9   Max Run Cadence    1883 non-null   float64       
     10  Total Ascent       1855 non-null   float64       
     11  Total Descent      1861 non-null   float64       
     12  Avg Stride Length  1883 non-null   float64       
     13  Min Temp           607 non-null    float64       
     14  Max Temp           607 non-null    float64       
     15  Min Elevation      1875 non-null   float64       
     16  Max Elevation      1880 non-null   float64       
     17  Total Run Time     1883 non-null   float64       
    dtypes: datetime64[ns](1), float64(16), object(1)
    memory usage: 279.5+ KB
    

Over half of the values in the *Avg HR*, *Max HR*, *Min Temp* and *Max Temp* columns are NULL. Remember, I'm doing this, so I can get a better understanding of trends in my running data over the years I ran in college. I wan't to create some visualizations with this data in the future, and all these values are important in getting a "big picture" look at my runnning trends. To fill in the missing data, we have four options: 

1. Drop the rows that are missing data. 
2. Fill NULL rows with some sort of common value (oftentimes the median of the column in question).
3. Bring in an outside data source.
4. Create a predictive model. 

Option 1 is not going to work here as that would eliminate half my data. Option 2 works OK for the columns that are missing only a few features, but it would definitely take away from the richness of the data and make for some boring / unhelpful visualizations if we used it for all the missing values in the dataset. But, option 3 can work great for filling in the tempurature data as it is easy to find weather data, and option 4 is the way to go for fixing the HR data!


```python
#Using Option 1 to infill missing data 
cols_with_few_nan = ['Total Ascent', 'Total Descent','Min Elevation', 'Max Elevation']
df[cols_with_few_nan] = df[cols_with_few_nan].fillna(df[cols_with_few_nan].median())
```

## Bringing in Some Outside Help

Unfortunately, Garmin uses a somewhat cryptic system to log the location of runs. It usually titles each activity as either the county or city name plus "Running" with no other geolocation data to go along with it. To help us get started, let's look at where my runs occured that are missing tempurature data.


```python
from collections import Counter

run_locations = Counter(df[df['Min Temp'].isna()]['Title'])
sorted(run_locations.items(), key=lambda x:x[1], reverse = True)[:10]
```




    [('Oktibbeha County Running', 349),
     ('Starkville Running', 245),
     ('Flowood Running', 127),
     ('Jackson County Running', 120),
     ('Moss Point Running', 64),
     ('Mobile County Running', 50),
     ('Boulder County Running', 43),
     ('Lucedale Running', 31),
     ('Oktibbeha County - Running', 21),
     ('Boulder Running', 21)]



Despite differing titles, the the vast majority of these samples occur either in my old college town of Starkville, MS or very close to it, and nearly all the rest occur somewhere in Mississippi or in the South. Because we don't have a way to convert these titles to a more specific location without getting really messy, I believe it will suffice to use weather data for Starkville, MS as a proxy for all the missing values we have. 


```python
df_weather = pd.read_csv("Weather Data.csv")
df_weather = df_weather[['NAME','DATE','TMAX','TMIN']].reset_index()
df_weather.drop(columns = 'index', inplace = True)

df_weather['Date'] = pd.to_datetime(df_weather['DATE'])
df_weather['Min Temp'] = df_weather['TMIN']
df_weather['Max Temp'] = df_weather['TMAX']

#Dataset contains weather reports from several locations surrounding Starkville, so we can group them together. 
df_weather = df_weather[['Date', 'Min Temp', 'Max Temp']].groupby(by = ['Date']).mean()

#Perform inner join, giving us a 1:1 ratio of dates to tempuratures
df = df.drop(columns = ['Min Temp', 'Max Temp']).merge(df_weather, on = 'Date', how = 'inner')

#Infill any remaining missing tempurature values with the median
cols_with_few_nan = ['Min Temp', 'Max Temp']
df[cols_with_few_nan] = df[cols_with_few_nan].fillna(df[cols_with_few_nan].median())
```

There you have it, our filled in tempurature data. Now, we need to build a model(s) that can effectively populate the missing values in our Max HR and Avg HR columns.

## Fitting a model to fill-in our missing data
Let's train and evaluate some regression models to fill in all that missing heart rate data. In the end we will have built two models, one to predict the *Avg HR* columns and another to predict *Max HR*.


```python
#Select subset of data with no missing values for training
df_train = df.dropna()

#Training features
X_train = df_train[['Avg Pace', 'Best Pace', 'Distance', 'Calories',
       'Avg Run Cadence', 'Max Run Cadence',
       'Total Ascent', 'Total Descent', 'Avg Stride Length', 'Min Elevation',
       'Max Elevation', 'Total Run Time', 'Min Temp', 'Max Temp']]


y_avg = df_train['Avg HR']
y_max = df_train['Max HR']
```

Becuase of my running domain knowledge, I have an idea of what features will be usefull for predicting the *Max HR* and *Avg HR* columns of our data, but I'm a fan of letting scikit-learn decide what features are best for me. Let's select the best 5 features. 


```python
from sklearn.feature_selection import SelectKBest, f_regression

#The best features to predict Avg HR are not necessarily the best to predict Max HR
kb_average = SelectKBest(f_regression, k=5).fit(X_train, y_avg)
kb_max = SelectKBest(f_regression, k=5).fit(X_train, y_max)

X_avg = kb_average.transform(X_train)
X_max = kb_max.transform(X_train)
```

Now we can use those extracted features to train several regression models and evaluate using cross validation to pick the best one for our two prediction tasks. The evaluation metrics we'll use are Mean Absolute Error (MAE) and Mean Squared Error (MSE).


```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error
import statistics

def cv(model, X, y, model_name):
    score = cross_validate(model, X, y, cv=5, scoring=('neg_mean_absolute_error', 'neg_mean_squared_error'))
    print("\nModel: ", model_name)
    print("Test Mean Absolute Error: ", statistics.mean(score['test_neg_mean_absolute_error']), 
          "\nTest Mean Sqaured Error: ", statistics.mean(score['test_neg_mean_squared_error']))

lasso = Lasso(alpha = 0.1)
reg = LinearRegression()
regr = SVR(C=1.0, epsilon=0.2)
rfr = RandomForestRegressor(max_depth=10)

data_list = [(X_avg, y_avg, "HR Avg"), (X_max, y_max, "HR Max")]
model_dict = {"Lasso Regression":lasso, "Linear Regression":reg, "SVR":regr, "RF Regressor":rfr}

for data in data_list:
    print('\n############################\n',data[2])
    for model in model_dict.keys():
        cv(model_dict[model], data[0], data[1], model)
```
    ############################
     HR Avg
    
    Model:  Lasso Regression
    Test Mean Absolute Error:  -3.4721680220384203 
    Test Mean Sqaured Error:  -21.528693269323934
    
    Model:  Linear Regression
    Test Mean Absolute Error:  -3.4965374931940674 
    Test Mean Sqaured Error:  -21.794449648893643
    
    Model:  SVR
    Test Mean Absolute Error:  -6.503686070502235 
    Test Mean Sqaured Error:  -64.17778768337854
    
    Model:  RF Regressor
    Test Mean Absolute Error:  -3.6666079473371216 
    Test Mean Sqaured Error:  -22.542642038967205
    
    ############################
     HR Max
    
    Model:  Lasso Regression
    Test Mean Absolute Error:  -6.310069863144789 
    Test Mean Sqaured Error:  -67.01035553775584
    
    Model:  Linear Regression
    Test Mean Absolute Error:  -6.332889188711009 
    Test Mean Sqaured Error:  -67.28914455377931
    
    Model:  SVR
    Test Mean Absolute Error:  -7.571264852476413 
    Test Mean Sqaured Error:  -91.18257365781352
    
    Model:  RF Regressor
    Test Mean Absolute Error:  -5.748229756457544 
    Test Mean Sqaured Error:  -57.09265923926437
    

Lasso regression is the best model for predicting the Average Heart Rate of my runs while Random Forest Regressor is the best at predicting the Max Heart Rate. 

## Fit models and Visualize Performance
Let's fit the best performing models using their entire respective training sets and predict on the samples that are missing HR data.


```python
lasso_avg = Lasso(alpha = 0.1)
lasso_avg.fit(X_avg, y_avg)

rfr_max = RandomForestRegressor(max_depth=10)
rfr_max.fit(X_max, y_max)

#Select predictive features from entire dataset
X_full = df[['Avg Pace', 'Best Pace', 'Distance', 'Calories',
       'Avg Run Cadence', 'Max Run Cadence',
       'Total Ascent', 'Total Descent', 'Avg Stride Length', 'Min Elevation',
       'Max Elevation', 'Total Run Time', 'Min Temp', 'Max Temp']].to_numpy()

#Predict for all samples and infill rows that are missing values
df['Max HR'] = df['Max HR'].combine_first(pd.Series(rfr_max.predict(kb_max.transform(X_full)).tolist()))
df['Avg HR'] = df['Avg HR'].combine_first(pd.Series(lasso_avg.predict(kb_average.transform(X_full)).tolist()))

df.to_csv('Running_Data_Clean.csv')
df
``` 
{{< rawhtml >}}
<div  style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Title</th>
      <th>Avg Pace</th>
      <th>Best Pace</th>
      <th>Distance</th>
      <th>Calories</th>
      <th>Avg HR</th>
      <th>Max HR</th>
      <th>Avg Run Cadence</th>
      <th>Max Run Cadence</th>
      <th>Total Ascent</th>
      <th>Total Descent</th>
      <th>Avg Stride Length</th>
      <th>Min Elevation</th>
      <th>Max Elevation</th>
      <th>Total Run Time</th>
      <th>Min Temp</th>
      <th>Max Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-12-31</td>
      <td>Sevierville Running</td>
      <td>6.566667</td>
      <td>5.983333</td>
      <td>13.62</td>
      <td>1462.0</td>
      <td>179.348833</td>
      <td>175.218101</td>
      <td>175.0</td>
      <td>187.0</td>
      <td>381.0</td>
      <td>425.0</td>
      <td>1.40</td>
      <td>958.0</td>
      <td>1181.0</td>
      <td>89.516667</td>
      <td>22.0</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-12-30</td>
      <td>Sevierville Running</td>
      <td>7.100000</td>
      <td>6.533333</td>
      <td>5.83</td>
      <td>631.0</td>
      <td>151.453714</td>
      <td>168.844806</td>
      <td>176.0</td>
      <td>191.0</td>
      <td>169.0</td>
      <td>9.0</td>
      <td>1.29</td>
      <td>1001.0</td>
      <td>1174.0</td>
      <td>41.383333</td>
      <td>27.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-12-29</td>
      <td>Sevierville Running</td>
      <td>6.166667</td>
      <td>5.350000</td>
      <td>8.22</td>
      <td>881.0</td>
      <td>163.870123</td>
      <td>176.137507</td>
      <td>176.0</td>
      <td>191.0</td>
      <td>285.0</td>
      <td>184.0</td>
      <td>1.47</td>
      <td>968.0</td>
      <td>1167.0</td>
      <td>50.750000</td>
      <td>28.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-12-29</td>
      <td>Sevierville Running</td>
      <td>6.950000</td>
      <td>6.316667</td>
      <td>1.97</td>
      <td>209.0</td>
      <td>139.974450</td>
      <td>164.446402</td>
      <td>174.0</td>
      <td>191.0</td>
      <td>48.0</td>
      <td>181.0</td>
      <td>1.34</td>
      <td>1028.0</td>
      <td>1178.0</td>
      <td>13.700000</td>
      <td>28.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-12-28</td>
      <td>Moss Point Running</td>
      <td>7.066667</td>
      <td>6.383333</td>
      <td>7.37</td>
      <td>797.0</td>
      <td>156.137539</td>
      <td>171.490540</td>
      <td>173.0</td>
      <td>185.0</td>
      <td>182.0</td>
      <td>195.0</td>
      <td>1.32</td>
      <td>64.0</td>
      <td>135.0</td>
      <td>52.066667</td>
      <td>25.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1878</th>
      <td>2022-01-05</td>
      <td>Starkville Running</td>
      <td>6.950000</td>
      <td>6.316667</td>
      <td>11.50</td>
      <td>1014.0</td>
      <td>147.000000</td>
      <td>163.000000</td>
      <td>178.0</td>
      <td>201.0</td>
      <td>420.0</td>
      <td>404.0</td>
      <td>1.30</td>
      <td>223.0</td>
      <td>402.0</td>
      <td>79.966667</td>
      <td>33.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>1879</th>
      <td>2022-01-04</td>
      <td>Noxubee County Running</td>
      <td>6.166667</td>
      <td>4.483333</td>
      <td>11.01</td>
      <td>851.0</td>
      <td>145.000000</td>
      <td>169.000000</td>
      <td>183.0</td>
      <td>232.0</td>
      <td>289.0</td>
      <td>246.0</td>
      <td>1.42</td>
      <td>130.0</td>
      <td>201.0</td>
      <td>67.800000</td>
      <td>30.0</td>
      <td>40.5</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>2022-01-03</td>
      <td>Starkville Running</td>
      <td>6.866667</td>
      <td>5.650000</td>
      <td>15.01</td>
      <td>1349.0</td>
      <td>148.000000</td>
      <td>171.000000</td>
      <td>179.0</td>
      <td>190.0</td>
      <td>807.0</td>
      <td>774.0</td>
      <td>1.31</td>
      <td>89.0</td>
      <td>243.0</td>
      <td>103.116667</td>
      <td>29.5</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>2022-01-02</td>
      <td>Leon County Running</td>
      <td>7.050000</td>
      <td>5.716667</td>
      <td>8.01</td>
      <td>688.0</td>
      <td>141.000000</td>
      <td>156.000000</td>
      <td>177.0</td>
      <td>188.0</td>
      <td>810.0</td>
      <td>978.0</td>
      <td>1.29</td>
      <td>-94.0</td>
      <td>167.0</td>
      <td>56.416667</td>
      <td>30.0</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>1882</th>
      <td>2022-01-01</td>
      <td>Tallahassee Running</td>
      <td>7.316667</td>
      <td>5.300000</td>
      <td>7.00</td>
      <td>593.0</td>
      <td>136.000000</td>
      <td>159.000000</td>
      <td>175.0</td>
      <td>186.0</td>
      <td>801.0</td>
      <td>863.0</td>
      <td>1.26</td>
      <td>-59.0</td>
      <td>205.0</td>
      <td>51.183333</td>
      <td>61.5</td>
      <td>79.0</td>
    </tr>
  </tbody>
</table>
<p>1883 rows × 18 columns</p>
</div>

{{< /rawhtml >}}

There you have it, a NULL-free clean dataset. We are dashboard ready now. Check out my next post to see what I can make with this in Tablaeu. 