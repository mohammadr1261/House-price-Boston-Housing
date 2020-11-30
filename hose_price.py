from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
# Read the data
X_full = pd.read_csv('E:\\kaggle\\home-data\\train.csv', index_col='Id')
X_test_full = pd.read_csv('E:\kaggle\home-data\\test.csv', index_col='Id')

# drop missed target
X_full.dropna(subset=['SalePrice'],inplace=True)
y=X_full.SalePrice
X_full.drop('SalePrice',axis=1,inplace=True)

#drop columns with more than 50% missed
miss_col=[col for col in X_full.columns if X_full[col].isnull().any()]
percent_missed=(X_full[miss_col].isnull().sum(axis=0)/X_full.shape[0]*100)
high_percent_missed=[percent_missed.index[i] for i in range(len(percent_missed)) if percent_missed.values[i]>50 ]
X_full.drop(high_percent_missed,axis=1,inplace=True)
X_test_full.drop(high_percent_missed,axis=1,inplace=True)

#drop seems unneccesary columns
X_bad_cat=[col for col in X_full.columns if X_full[col].dtype=='object' and  X_full[col].nunique()>=10]
X_full.drop(X_bad_cat,axis=1,inplace=True)
X_test_full.drop(X_bad_cat,axis=1,inplace=True)


X_cat=[col for col in X_full.columns if X_full[col].dtype=='object' and  X_full[col].nunique()<10]
X_num=[col for col in X_full.columns if X_full[col].dtype in ['int64','float64']]

# use pipe for prossesing
num_imputer=SimpleImputer(strategy="most_frequent")
cat_imputer=Pipeline(steps=[
    ('imputer',SimpleImputer( strategy="most_frequent")),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))])

prepocess=ColumnTransformer(transformers=[
    ('num',num_imputer,X_num),
    ('cat',cat_imputer,X_cat)
])

#this model used for initial model
#model = RandomForestRegressor()
#if you want train your model you must use incomment above line
model = RandomForestRegressor(max_features=18, n_estimators=24)

pipe=Pipeline(steps=[
    ('prepocess',prepocess),
    ('model',model)
])
pipe.fit(X_full,y)
preds_test=pipe.predict(X_test_full)

param_grid = [
{'model__n_estimators': [23,24,25,26], 'model__max_features': [16,17,18,19]},
{'model__bootstrap': [False], 'model__n_estimators': [3, 10], 'model__max_features': [3,5,7]},
]

# print(pipe.get_params().keys())


# grid_search = GridSearchCV(
#     pipe, param_grid, cv=5,
#     scoring='neg_mean_squared_error',return_train_score=True
# )
# grid_search.fit(X_full,y)
# print(grid_search.best_score_)
# print(grid_search.best_params_)

#evaluate your model
# scores = -1 * cross_val_score(
#     pipe, X_full, y,cv=5,scoring='neg_mean_absolute_error')
# print("MAE mean score:")
# print(scores.mean())


output = pd.DataFrame({'Id': X_test_full.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
