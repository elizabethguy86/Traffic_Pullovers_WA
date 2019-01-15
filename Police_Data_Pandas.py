
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import statsmodels.discrete.discrete_model as sm


data_raw = pd.read_csv('/home/ec2-user/WA-clean.csv.gz', compression='gzip')
df = data_raw.drop(['state', 'stop_time', 'location_raw', 'county_fips', 'police_department', 'driver_age_raw', 'driver_race_raw',
                         'violation_raw','search_type_raw', 'is_arrested'], axis=1)
df['stop_date'] = pd.to_datetime(df.stop_date) #make stopdate a datetime object
df.driver_age.fillna(df.driver_age.mean(), inplace=True) #fill missing driver ages with mean

#Dummy coding:
df['driver_gender'] = pd.Series(np.where(df.driver_gender.values == 'F', 1, 0),
          df.index)
df['officer_gender'] = pd.Series(np.where(df.officer_gender.values == 'F', 1, 0),
          df.index)
race_dummies = pd.get_dummies(df.driver_race)
officer_race = pd.get_dummies(df.officer_race)
officer_race.columns = ['O_Asian', 'O_Black', 'O_Hispanic', 'O_Other', 'O_White']
merged = df.merge(race_dummies, left_index=True, right_index=True)
merged = merged.merge(officer_race, left_index=True, right_index=True)
merged['drugs_related_stop'] = pd.Series(np.where(merged.drugs_related_stop.values == False, 0, 1),
          merged.index)

#was a search conducted --> This is the outcome variable
merged['search_conducted'] = pd.Series(np.where(merged.search_conducted.values == False, 0, 1),
          merged.index)

merged['White_White'] = merged.White * merged.O_White#White driver White officer
merged['Black_White'] = merged.Black * merged.O_White#Black driver White officer
merged['Asian_White'] = merged.Asian * merged.O_White#Asian driver White officer
merged['Hispanic_White'] = merged.Hispanic * merged.O_White#Hispanic driver White officer
merged['White_Black'] = merged.White * merged.O_Black#White driver Black officer
merged['Black_Black'] = merged.Black * merged.O_Black #Black driver Black officer

X = merged.loc[:, ['driver_gender', 'driver_age', 'officer_gender', 'drugs_related_stop', 'Asian', 'Black', 'Hispanic',
                   'Other', 'White', 'O_Asian', 'O_Black', 'O_Hispanic', 'O_Other',
                   'O_White', 'White_White', 'Black_White', 'Asian_White',
                   'Hispanic_White', 'White_Black', 'Black_Black']]

y = merged.loc[:, ['search_conducted']]
y = y.values.reshape(8624032,)

fitted_X_train, fitted_X_test, fitted_y_train, fitted_y_test = train_test_split(X, y)

log_model = LogisticRegression()

log_model.fit(fitted_X_train, fitted_y_train)

preds = log_model.predict_proba(fitted_X_test)
log_loss(fitted_y_test, preds)
#log_loss = 0.091707177832494116


#Outputs:

'''[('driver_gender', -0.45287198553218022),
 ('driver_age', -0.022968236778229147),
 ('officer_gender', -0.010071773580308693),
 ('drugs_related_stop', 5.3455525078345518),
 ('Asian', 4.4184342895010786),
 ('Black', 5.0817284617090754),
 ('Hispanic', 4.9672820609552959),
 ('Other', 5.6290894859918872),
 ('White', 4.51373788931362),
 ('O_Asian', -1.02323305170735),
 ('O_Black', -1.3383925344273988),
 ('O_Hispanic', -1.2568753572088545),
 ('O_Other', -1.2627456694534289),
 ('O_White', -1.2557085389982203),
 ('White_White', 0.15454493742075387),
 ('Black_White', 0.28540365322816114),
 ('Asian_White', 0.0098635316063967592),
 ('Hispanic_White', 0.16441330211105804),
 ('White_Black', 0.0030278565889577521),
 ('Black_Black', 0.23015004098785694)]'''