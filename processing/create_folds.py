import numpy as np 
import pandas as pd
import os
from sklearn import model_selection
import config
def create_folds(data):
	target=config.response_variable
	# we create a new column called kfold and fill it with -1 data["kfold"] = -1
	# the next step is to randomize the rows of the data
	data = data.sample(frac=1).reset_index(drop=True)
	# calculate the number of bins by Sturge's rule # I take the floor of the value, you can also
	# just round it
	num_bins = int(np.floor(1 + np.log2(len(data))))
	# bin targets
	data.loc[:, "bins"] = pd.cut(data[target], bins=num_bins, labels=False)
	# initiate the kfold class from model_selection module
	kf = model_selection.StratifiedKFold(n_splits=5)
	# fill the new kfold column
	# note that, instead of targets, we use bins!
	for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
		data.loc[v_, 'kfold'] = f
	# drop the bins column
	data = data.drop("bins", axis=1) # return dataframe with folds return data
	return data

if __name__ == "__main__":
	path=config.train_file
	
	output_path=config.data_folder
	df=pd.read_csv(path)
	df = create_folds(df)
	df.to_csv(os.path.join(output_path,'bank_train_folds.csv'),index=False)
