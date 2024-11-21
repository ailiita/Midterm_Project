import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import pickle

# ------------------- Model Parameters -------------------------------------
random_state = 1
n_estimators = 3
max_depth = 6

output_file = 'model_rf.bin' 

n_splits = 5

# Data preparation ------------------------------------------------------------
df = pd.read_csv('AQ_data_cleaned.csv', delimiter=',')
del df['Unnamed: 0']

# Splitting data --------------------------------------------------------------
df_full_train, df_test = train_test_split(df, test_size=0.2, shuffle = False)
df_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y = df_full_train.aqi_class
y_train = df_train.aqi_class
y_test = df_test.aqi_class

del df_full_train['aqi_class']
del df_train['aqi_class']
del df_test['aqi_class']

dico = df_full_train.to_dict(orient='records')
train_dicts = df_train.to_dict(orient='records')
train_test = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dico)
X_train = dv.fit_transform(train_dicts)
X_test = dv.transform(train_test)

features = dv.get_feature_names_out()

# Training ------- ------------------------------------------------------------

# Validation   ------------------------------------------------------------
print(f'Doing cross-validation with n_estimators = {n_estimators}, max_depth = {max_depth} ')

rf = RandomForestClassifier(n_estimators=3, max_depth=6,random_state=1)

tscv = TimeSeriesSplit(n_splits=8)
cv_scores = []
i=1
for train_index, test_index in tscv.split(X):
    
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    rf.fit(X_train_cv, y_train_cv)
    score = rf.score(X_test_cv, y_test_cv)  # or use other metrics
    cv_scores.append(score)
    print(f'accuracy on fold {i} is {score} ')
    i+=1


print('Cross-validation results : ')
print(' %.3f +- %.3f' % ( np.mean(cv_scores), np.std(cv_scores)))

# Train Final model --------------------------------------------------------
print('Training final model')

model = RandomForestClassifier(n_estimators = n_estimators, 
                                   max_depth = max_depth, 
                                   random_state=random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f' Validation accuracy : {accuracy}')


# Save the model -----------------------------------------------------------

with open(output_file, 'wb') as f_out :
    pickle.dump((dv,model), f_out)
    
print(f'Model is saved to {output_file}')

