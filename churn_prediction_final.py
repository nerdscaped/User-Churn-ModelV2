# %pip install imblearn
# %pip install xgboost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score, mean_squared_error, roc_auc_score
from imblearn.combine import SMOTEENN
import math  
from scipy import stats
from io import StringIO
from xgboost import XGBClassifier
import boto3
from datetime import datetime
import time 
from collections import Counter

data = pd.read_csv('/Users/mattcadel/Documents/Python/DSML/churn_prediction.csv')

columns = data.columns
data.to_parquet('df.parquet.gzip',compression='gzip')
del data
data = pd.read_parquet('df.parquet.gzip', columns=columns)

data['avg_hour_L60D'] = round((math.pi * (data['avg_hour_L60D'] + 1) / 24).apply(math.cos), 2)
data['month_access_date'] = round((math.pi * data['month_access_date'] / 12).apply(math.cos), 2)

X = data[['user_type', 'days_last_access',
       'month_access_date', 'days_first_access',
       'first_brand_ranking_indicator', 'plays', 't20_plays', 'actions',
       'plays_L60D', 't20_plays_L60D', 'recs_L60D', 'actions_L60D',
       'brands_played_L60D', 'subcats_played_L60D', 'platforms_L60D',
       'weeks_accessed_L60D', 'unq_recs_L60D', 'avg_hour_L60D', 'plays_L7D',
       't20_plays_L7D', 'recs_L7D', 'day_bounce_rate_L7D', 'brands_played_L7D',
       'actions_L7D', 'days_accessed_L7D', 'plays_delta', 't20_plays_delta',
       'recs_delta', 'actions_delta', 'day_bounce_rate_delta',
       'brands_played_delta', 'subcats_played_delta']]
y = data[['churn_status', 'user_type']]

users = data.loc[(data['user_type'] == 'predict')][['user_primaryid', 'plays_L60D']].reset_index()[['user_primaryid', 'plays_L60D']]
del data

X_train = X.loc[(X['user_type'] == 'train')]
X_train.pop('user_type')
y_train = y.loc[(y['user_type'] == 'train')]
y_train.pop('user_type')
X_predict = X.loc[(X['user_type'] == 'predict')]
X_predict.pop('user_type')
y_predict = y.loc[(y['user_type'] == 'predict')]
y_predict.pop('user_type')

del X
del y

##### Sample Data (as data is imbalanced)

# In testing, SMOTEENN sampling was identified as being the best sampler

resampler = SMOTEENN(random_state = 0)

X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

binary_X_train = X_train['first_brand_ranking_indicator']
binary_X_test = X_test['first_brand_ranking_indicator']
binary_predict_X = X_predict['first_brand_ranking_indicator']

continuous_X_train = X_train[['days_last_access',
       'month_access_date', 'days_first_access','plays', 't20_plays', 'actions',
       'plays_L60D', 't20_plays_L60D', 'recs_L60D', 'actions_L60D',
       'brands_played_L60D', 'subcats_played_L60D', 'platforms_L60D',
       'weeks_accessed_L60D', 'unq_recs_L60D', 'avg_hour_L60D', 'plays_L7D',
       't20_plays_L7D', 'recs_L7D', 'day_bounce_rate_L7D', 'brands_played_L7D',
       'actions_L7D', 'days_accessed_L7D', 'plays_delta', 't20_plays_delta',
       'recs_delta', 'actions_delta', 'day_bounce_rate_delta',
       'brands_played_delta', 'subcats_played_delta']]

continuous_X_test = X_test[['days_last_access',
       'month_access_date', 'days_first_access','plays', 't20_plays', 'actions',
       'plays_L60D', 't20_plays_L60D', 'recs_L60D', 'actions_L60D',
       'brands_played_L60D', 'subcats_played_L60D', 'platforms_L60D',
       'weeks_accessed_L60D', 'unq_recs_L60D', 'avg_hour_L60D', 'plays_L7D',
       't20_plays_L7D', 'recs_L7D', 'day_bounce_rate_L7D', 'brands_played_L7D',
       'actions_L7D', 'days_accessed_L7D', 'plays_delta', 't20_plays_delta',
       'recs_delta', 'actions_delta', 'day_bounce_rate_delta',
       'brands_played_delta', 'subcats_played_delta']]

continuous_predict_X = X_predict[['days_last_access',
       'month_access_date', 'days_first_access','plays', 't20_plays', 'actions',
       'plays_L60D', 't20_plays_L60D', 'recs_L60D', 'actions_L60D',
       'brands_played_L60D', 'subcats_played_L60D', 'platforms_L60D',
       'weeks_accessed_L60D', 'unq_recs_L60D', 'avg_hour_L60D', 'plays_L7D',
       't20_plays_L7D', 'recs_L7D', 'day_bounce_rate_L7D', 'brands_played_L7D',
       'actions_L7D', 'days_accessed_L7D', 'plays_delta', 't20_plays_delta',
       'recs_delta', 'actions_delta', 'day_bounce_rate_delta',
       'brands_played_delta', 'subcats_played_delta']]

del X_predict
del X_test
del X_train

##### Normalising Data

gaussian_features = []
non_gaussian_features = []

# Testing for Normality for Standisation
for i in continuous_X_train.columns:
    data = list(continuous_X_train[i][:5000])
    normality_tests = {'Shapiro-Wilk Test': stats.shapiro(data)}
    
    for test_name, test_result in normality_tests.items():
        p_value = test_result[1]
        alpha = 0.05  # Significance level
        if p_value < alpha:
            non_gaussian_features.append(i)
        else:
            gaussian_features.append(i)

continuous_X_train_for_transformer = continuous_X_train[non_gaussian_features] # For Non-Gaussian Features
continuous_X_train_for_scaler = continuous_X_train[gaussian_features] # For Gaussian Features

continuous_X_test_for_transformer = continuous_X_test[non_gaussian_features] # For Non-Gaussian Features
continuous_X_test_for_scaler = continuous_X_test[gaussian_features] # For Gaussian Features

continuous_predict_X_for_transformer = continuous_predict_X[non_gaussian_features] # For Non-Gaussian Features
continuous_predict_X_for_scaler = continuous_predict_X[gaussian_features] # For Gaussian Features

quantile_transformer = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
quantile_transformer.fit(continuous_X_train_for_transformer)

continuous_X_train_for_transformer = quantile_transformer.transform(continuous_X_train_for_transformer)
continuous_X_test_for_transformer = quantile_transformer.transform(continuous_X_test_for_transformer)
continuous_predict_X_for_transformer = quantile_transformer.transform(continuous_predict_X_for_transformer)

continuous_X_train_for_transformer = pd.DataFrame(continuous_X_train_for_transformer, index=continuous_X_train.index, columns=non_gaussian_features)
continuous_X_test_for_transformer = pd.DataFrame(continuous_X_test_for_transformer, index=continuous_X_test.index, columns=non_gaussian_features)
continuous_predict_X_for_transformer = pd.DataFrame(continuous_predict_X_for_transformer, index=continuous_predict_X.index, columns=non_gaussian_features)

X_train = pd.concat([binary_X_train, continuous_X_train_for_transformer], axis=1)
del continuous_X_train_for_transformer
X_test = pd.concat([binary_X_test, continuous_X_test_for_transformer], axis=1)
del continuous_X_test_for_transformer
X_predict = pd.concat([binary_predict_X, continuous_predict_X_for_transformer], axis=1)
del continuous_predict_X_for_transformer

scaler = StandardScaler()
if len(gaussian_features) > 0:
    scaler.fit(continuous_X_train_for_scaler)
    continuous_X_train_for_scaler = scaler.transform(continuous_X_train_for_scaler)
    continuous_X_test_for_scaler = scaler.transform(continuous_X_test_for_scaler)
    continuous_predict_X_for_scaler = scaler.transform(continuous_predict_X_for_scaler)

    continuous_X_train_for_scaler = pd.DataFrame(continuous_X_train_for_scaler, index=continuous_X_train.index, columns=gaussian_features)
    continuous_X_test_for_scaler = pd.DataFrame(continuous_X_test_for_scaler, index=continuous_X_test.index, columns=gaussian_features)
    continuous_predict_X_for_scaler = pd.DataFrame(continuous_predict_X_for_scaler, index=continuous_predict_X.index, columns=gaussian_features)

    X_train = pd.concat([X_train, continuous_X_train_for_scaler], axis=1)
    del continuous_X_train_for_scaler
    X_test = pd.concat([X_test, continuous_X_test_for_scaler], axis=1)
    del continuous_X_test_for_scaler
    X_predict = pd.concat([X_predict, continuous_predict_X_for_scaler], axis=1)
    del continuous_predict_X_for_scaler
else: 
    print('No Gaussian Features Found, Continuing to Model Training')

# Running ML Model
model = XGBClassifier(objective='binary:logistic', max_depth=5)

model.fit(X_train, y_train)
    
# Review model
predictions_test = model.predict(X_test)
      
print('roc_auc_score\n',roc_auc_score(y_test, predictions_test))
print('mean_squared_error\n',mean_squared_error(y_test, predictions_test))
print('balanced_accuracy_score\n',balanced_accuracy_score(y_test, predictions_test))
print('recall_score\n',recall_score(y_test, predictions_test))

# Preductions Output
predictions = model.predict(X_predict)
predictions_test_prob = model.predict_proba(X_predict)
predictions_test_prob = [x[1] for x in predictions_test_prob]
dates = [datetime.today().strftime('%Y-%m-%d')] * len(predictions_test_prob)

predictions = pd.DataFrame.from_dict({'churn_prediction': predictions,
                                       'churn_probability': predictions_test_prob, 
                                       'date':dates})

predictions = pd.concat([users, predictions], axis=1)
print('predictions\n', predictions)

predictions.to_csv('/Users/mattcadel/Documents/Python/DSML/user_id_predictions.csv')