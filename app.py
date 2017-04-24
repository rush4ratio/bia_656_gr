#!flask/bin/python
from flask import Flask
import warnings
warnings.filterwarnings('ignore')
import pickle
import pandas as pd
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
from flask import request

os.chdir("final_model")


filename='rf_model_wo_text_features'

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

os.chdir("..")


X_feat=None



app = Flask(__name__)

type_of_pe = {'Parastatal':7,
         'University':9, 'Hospital':4, 'Schools And Colleges': 8,
       'Ministry':6, 'Local Authorities':5, 'Co-operative Society':1, 'Banks':0,
       'County Assembly':2, 'County Government':3}

proc_method = {'Open tender':0,'Restricted Tender':1}

def advert_date_month_conversion(x):
    return x.to_datetime().month


@app.route('/prediction', methods=['POST'])
def get_prediction():
  type_of_procuring_entity=[request.form['type_of_pe']]
  procuring_method=[request.form['proc_method']]
  advert_date=[request.form['advert_date']]
  tenders_sold=[int(request.form['tenders_sold'])]
  bids_received=[int(request.form['bids_received'])]

  one_contract = {'bids_received':bids_received,'tenders_sold':tenders_sold, 
'type_of_procuring_entity':type_of_procuring_entity
                ,'procuring_method':procuring_method,'advert_date':advert_date}

  one_contract = pd.DataFrame.from_dict(one_contract)

  one_contract['advert_date'] = one_contract['advert_date'].apply(parse)
  one_contract['ad_date_months'] = one_contract.advert_date.apply(advert_date_month_conversion)


  one_contract['procuring_method_enc'] = proc_method[one_contract.procuring_method.values[0]]
  one_contract['type_of_procuring_entity_enc'] = type_of_pe[one_contract.type_of_procuring_entity.values[0]]


  # proc_ent
  proc_ent_dict={}

  for i in range(1, 9+1):
      proc_ent_dict['proc_ent_' + str(i)] = [0]


  if one_contract.type_of_procuring_entity_enc.values[0] != 0:
      for key in proc_ent_dict.keys():
          if str(one_contract.type_of_procuring_entity_enc.values[0]) in key:
               proc_ent_dict[key] = [1]

  proc_ent_dict = pd.DataFrame.from_dict(proc_ent_dict)

  one_contract = pd.concat([one_contract,proc_ent_dict], axis =1)


  # ad_date_months
  ad_mth_dict={}
  for i in range(2, 12+1):
      ad_mth_dict['ad_mnth_' + str(i)] = [0]

  if one_contract.ad_date_months.values[0] != 1:
      for key in ad_mth_dict.keys():
          if str(one_contract.ad_date_months.values[0]) in key:
               ad_mth_dict[key] = [1]

  ad_mth_dict = pd.DataFrame.from_dict(ad_mth_dict)

  one_contract = pd.concat([one_contract,ad_mth_dict], axis =1)



  # description_data = extract_text_features(one_contract.description)

  one_contract = one_contract.drop(['advert_date','procuring_method','type_of_procuring_entity','type_of_procuring_entity_enc','ad_date_months'], axis=1)


  X_feat = (one_contract)

  result = loaded_model.predict(X_feat)
  
  prediction = "Prediction: KES. " + str(result[0])

  
  return '<!DOCTYPE html><html><head><link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.css"><title>Decision Support System</title></head><body class="container" style="padding-top: 50px;"><h1>'+prediction +'</h1><a href="/" class="button button-primary  ">BACK</a></body></html>'

@app.route('/')
def index():
        return '''
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.css">
  <title>Decision Support System</title>
</head>
<body class="container" style="padding-top: 50px;">
<form action="http://localhost:5000/prediction" method="post">
<h2>Estimate contract awards by providing details below:</h2>
<hr / >
  <div class="row">
    <div class="six columns">
      <label for="type_of_pe">Type of Procuring Entity</label>
      <select class="u-full-width" name="type_of_pe" id="type_of_pe">
    <option value="Banks">Banks</option>
    <option value="Co-operative Society">Co-operative Society</option>
    <option value="County Assembly">County Assembly</option>
    <option value="County Government">County Government</option>
    <option value="Hospital">Hospital</option>
    <option value="Local Authorities">Local Authorities</option>
    <option value="Ministry">Ministry</option>
    <option value="Parastatal">Parastatal</option>
    <option value="Schools And Colleges" >Schools And Colleges</option>
    <option value="University" >University</option>
    </select>
    </div>
  </div>
  <div class="row">
    <div class="six columns">
      <label for="proc_method">Procuring Method: </label>
      <select name="proc_method" class="u-full-width" id="proc_method">
    <option value="Open tender">Open Tender</option>
    <option value="Restricted Tender">Restricted Tender</option>
  </select>
    </div>
  </div>
  <div class="row">
    <div class="six columns">
      <label for="advert_date">Advert Date: </label>
        <input name="advert_date" class="u-full-width" type="date" id="advert_date">
    </div>
  </div>
  <div class="row">
    <div class="six columns">
      <label for="tenders_sold">Tenders Sold: </label>
        <input name="tenders_sold" class="u-full-width" type="number" id="tenders_sold">
    </div>
  </div>
  <div class="row">
    <div class="six columns">
      <label for="bids_received">Bids Received: </label>
        <input name="bids_received" class="u-full-width" type="number" id="bids_received">
    </div>
  </div>
  <input class="button-primary" type="submit" value="Submit">
</form>

</body>
</html>

'''
if __name__ == '__main__':
    app.run(debug=True)
