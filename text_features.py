import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

def extract_text_features(description):
	description=description_data.apply(lambda x: x.replace("\r","").replace("\n","").replace("\t",""))
	tfv = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word',\
token_pattern=r'\w{1,}', ngram_range=(1,2), use_idf = 1, smooth_idf = 1, sublinear_tf = 1, stop_words='english')

	description_data = tfv.fit_transform(list(description))


	# Truncate 2,360 text features to 120 components
	svd = TruncatedSVD(n_components = 120,random_state=123)
	svd.fit(description_data)

	description_data = svd.transform(description_data)

	return description_data
