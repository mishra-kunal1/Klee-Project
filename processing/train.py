import joblib
import pandas as pd
from sklearn import metrics 
from sklearn import tree
import config
import argparse
import os
import model_dispatcher
def run(fold,model):
	# read the training data with folds
	path=config.fold_file
    
	response=config.response_variable
	df = pd.read_csv(path)
	# training data is where kfold is not equal to provided fold # also, note that we reset the index
	df_train = df[df.kfold != fold].reset_index(drop=True)
	# validation data is where kfold is equal to provided fold
	df_valid = df[df.kfold == fold].reset_index(drop=True)
	# drop the label column from dataframe and convert it to # a numpy array by using .values.
	# target is label column in the dataframe
	x_train = df_train.drop(response, axis=1).values
	y_train = df_train[response].values
	# similarly, for validation, we have
	x_valid = df_valid.drop(response, axis=1).values 
	y_valid = df_valid[response].values
	# initialize simple decision tree classifier from sklearn
	clf=model_dispatcher.models[model]
	#clf = tree.DecisionTreeClassifier()
	# fit the model on training data
	clf.fit(x_train, y_train)
	# create predictions for validation samples
	preds = clf.predict(x_valid)
	# calculate & print accuracy
	accuracy = metrics.accuracy_score(y_valid, preds) 
	print(f"Fold={fold}, Accuracy={accuracy}")
	# save the model
	joblib.dump(clf,os.path.join(config.model_folder ,f"dt_{fold}.bin"))
if __name__ == "__main__": 
	parser=argparse.ArgumentParser()
	parser.add_argument("--fold",type=int)
	parser.add_argument("--model",type=str)
	args=parser.parse_args()
	run(fold=args.fold, model=args.model)
  