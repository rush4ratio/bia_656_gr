import pickle
import pandas as pd
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from text_features import extract_text_features

def advert_date_month_conversion(x):
    return x.to_datetime().month








filename='rf_model'

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

X_feat=None
"""
Procurement Ref No  KIRDI 12/2013/2014
Procurement Method  Open tender
Contractor  Plan and Place Insurance Brokers Ltd
Units   0
Amount  KSH19,538,935.00
Advert Date 26th, June 2014
Notification Date   11th, July 2014
Contract Signing    1st, August 2014
Completion Date 31st, August 2015
Tenders Sold    16
Bids Recieved   16


Banks                     0                                  2
Co-operative Society      1                                  6
County Assembly           2                                 12
County Government         3                                  1
Hospital                  4                                138
Local Authorities         5                                 11
Ministry                  6                                137
Parastatal                7                               1924
Schools And Colleges      8                                 31
University                9                                146

"""

#data.groupby(['type_of_procuring_entity','type_of_procuring_entity_enc']).size()
type_of_pe = {'Parastatal':7,
         'University':9, 'Hospital':4, 'Schools And Colleges': 8,
       'Ministry':6, 'Local Authorities':5, 'Co-operative Society':1, 'Banks':0,
       'County Assembly':2, 'County Government':3}

# open tender 0 ; restricted tender 1
proc_method = {'Open tender':0,'Restricted Tender':1}





description = ['Provision of Inpatient Medical Insurance Cover']
type_of_procuring_entity = ['Parastatal']
procuring_method = ['Open tender']
advert_date = ['6/26/2014']
tenders_sold = [16]
bids_received = [15]

one_contract = {'description':description ,'type_of_procuring_entity':type_of_procuring_entity
                ,'procuring_method':procuring_method,'advert_date':advert_date
                ,'tenders_sold':tenders_sold, 'bids_received':bids_received}

one_contract = pd.DataFrame.from_dict(one_contract)

one_contract['advert_date'] = one_contract['advert_date'].apply(parse)
one_contract['ad_date_months'] = one_contract.advert_date.apply(advert_date_month_conversion)


one_contract['procuring_method_enc'] = proc_method[one_contract.procuring_method.values[0]]
one_contract['type_of_procuring_entity_enc'] = type_of_pe[one_contract.type_of_procuring_entity.values[0]]



# proc_ent

proc_ent_dict={}

for i in range(1, 9+1):
    proc_ent_dict['proc_ent_' + str(i)] = 0

if one_contract.type_of_procuring_entity_enc.values[0] != 0:
    for key in proc_ent_dict.keys():
        if str(one_contract.type_of_procuring_entity_enc.values[0]) in key:
             proc_ent_dict[key] = [1]


proc_ent_dict = pd.DataFrame.from_dict(proc_ent_dict)

one_contract = pd.concat([one_contract,proc_ent_dict], axis =1)

description_data = extract_text_features(one_contract.description)


X_feat = np.hstack((one_contract.drop('description', axis=1).as_matrix(),description_data))


result = loaded_model.predict(X_feat)
print(result)
