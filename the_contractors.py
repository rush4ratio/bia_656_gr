import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Custom modules
from impute_missing_data import impute_missing_vars
from text_features import extract_text_features
from model import model_evaluation,model_performance




def merge_ds():
	contracts = pd.read_csv("data/contract_awards-20161229-17_41_49.csv")
	procuring_entities = pd.read_csv("data/procuring-entities-20161230-19_07_58.csv")
	procuring_entities = procuring_entities.drop("rec", axis = 1)
	merged_ds  = contracts.merge(procuring_entities, left_on="procuring_entity", right_on="name", how="left")
	merged_ds = merged_ds.drop(["page", "name"], axis  = 1)
	merged_ds = merged_ds.rename(columns={'type':'type_of_procuring_entity'})

	return merged_ds

def comma_sep_to_float(num):
    return float(num.strip().replace(",",""))

def return_outliers_via_IQR(series):
    q1 = series.quantile(.25)
    q3 = series.quantile(.75)
    iqr = q3 - q1

    return (series < q1 - (1.5 * iqr)) | (series > q3 + (1.5 * iqr))

def advert_date_month_conversion(x):
    return x.to_datetime().month


def main():
	contracts = merge_ds()
	contracts.amount = contracts.amount.apply(comma_sep_to_float)
	date_cols = ['advert_date', 'notification_date','contract_signing','completion_date']
	for date_col in date_cols:
	    contracts[date_col] = contracts[date_col].apply(parse)

	# We don't need procurement ref nos
	contracts = contracts.drop('procurement_ref_no', axis = 1)
	cols = ['procuring_entity','description','contractor','type_of_procuring_entity','procuring_method']
	for col in cols:
	    contracts[col] = contracts[col].apply(lambda x: x.strip())

	# We are interested in open tenders and restricted tenders
	mask = (contracts['procuring_method'] == 'Open tender') | (contracts['procuring_method'] == 'Restricted Tender')
	contracts = contracts[mask]
	contracts.reset_index(drop=True, inplace=True)

	# Drop columns we don't need
	contracts = contracts.drop(['contractor','contract_signing', 'completion_date'],axis = 1)
	contracts = contracts.drop(['notification_date'], axis = 1)


	contracts['ad_date_months'] = contracts.advert_date.apply(advert_date_month_conversion)


	# Dealing with missing data
	data = contracts

	#Replace bids_received > tenders_sold
	data.ix[data.tenders_sold < data.bids_received ,['tenders_sold','bids_received']] = np.nan

	#Replace tenders_sold <0
	data.ix[data.tenders_sold <0 ,['tenders_sold','bids_received']] = np.nan;

	# Handle 0 case
	data['tenders_sold'] = data['tenders_sold'].replace({ 0 : np.nan })
	data['bids_received'] = data['bids_received'].replace({ 0 : np.nan })


	# Encode label vars
	lb = LabelEncoder()
	data['procuring_method_enc'] = lb.fit_transform(data.procuring_method)
	data['type_of_procuring_entity_enc'] = lb.fit_transform(data.type_of_procuring_entity)

	# Custom module to deal with missing 'bids received' and 'tenders sold'
	data = impute_missing_vars(data)
	contracts_type_pe = pd.get_dummies(data=data.type_of_procuring_entity_enc, prefix="proc_ent")

	# correct for degrees of freedom problem
	contracts_type_pe = contracts_type_pe.drop('proc_ent_0', axis =1)

	cols_to_drop = ['type_of_procuring_entity_enc','type_of_procuring_entity','advert_date','procuring_method','procuring_entity']
	contracts  = pd.concat([data.drop(cols_to_drop,axis=1), contracts_type_pe], axis =1)


	#get text features
	description_data = extract_text_features(contracts.description)


	# Stack everything together
	combined_features = np.hstack((contracts.drop('amount',axis=1).drop('description', axis=1).as_matrix(),description_data))
	X = combined_features
	Y = contracts.amount.as_matrix()

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=123)
	model_performance(X_test,Y_test)

if __name__ == "__main__": main()
