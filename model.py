# from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# getting the split

def model_evaluation(X,Y):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=123)

	# X_train, Y_train = X[list(train_indices)], Y[list(train_indices)]


	param_grid = [{'bootstrap': [False], 'n_estimators': [50,100], 'max_features':['sqrt',0.2]},]

	rf_regressor= RandomForestRegressor(criterion='mae',n_jobs=-1,warm_start=True,min_samples_leaf=50,random_state=123)



	grid_search.fit(X_train, Y_train)

	cvres = grid_search.cv_results_
	for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    	print(-mean_score, params)

def model_performance(X_test, Y_test):
    # 76022802.645 {'bootstrap': False, 'max_features': 0.2, 'n_estimators': 100}

    rf_regressor= RandomForestRegressor(criterion='mae',n_jobs=-1,warm_start=True,min_samples_leaf=50,random_state=123)



    grid_search.fit(X_test, Y_test)

