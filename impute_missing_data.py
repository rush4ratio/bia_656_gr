import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder


def impute_missing_vars(data):
	#---- Imputing missing data ------
	np.random.seed(123) # get constant vals across the board
	is_test = np.random.uniform(0, 1, len(data)) > 0.70
	train = data[is_test==False]
	test = data[is_test==True]

	tender_imputer = KNeighborsRegressor(n_neighbors=5)
	bids_imputer = KNeighborsRegressor(n_neighbors=5)

	#Split the training data set 
	train_w = train[train.tenders_sold.isnull()==False]
	train_w_null = train[train.tenders_sold.isnull()==True]

	#Create features
	cols= ['amount', 'ad_date_months','procuring_method_enc','type_of_procuring_entity_enc']

	tender_imputer.fit(train_w[cols], train_w.tenders_sold)

	tenders_sold_new = tender_imputer.predict(train_w_null[cols])
	bids_imputer.fit(train_w[cols], train_w.bids_received)
	bids_received_new = bids_imputer.predict(train_w_null[cols])

	#Replace the missing value in training data
	#Round to nearest integer
	train_w_null['tenders_sold'] = np.rint(tenders_sold_new)
	train_w_null['bids_received'] = np.rint(bids_received_new)


	test_w_null = test[test.tenders_sold.isnull()==True]
	test_w = test[test.tenders_sold.isnull()==False]

	tenders_sold_test = tender_imputer.predict(test_w_null[cols])
	bids_received_test = bids_imputer.predict(test_w_null[cols])

	test_w_null['tenders_sold'] = np.rint(tenders_sold_test)
	test_w_null['bids_received'] = np.rint(bids_received_test)

	#combine the data back together
	#Train dataset
	train = train_w.append(train_w_null)
	#Test dataset
	test = test_w.append(test_w_null)

	data2= train.append(test)

	return data2



