from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
models = {
"decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini" ),
"decision_tree_entropy": tree.DecisionTreeClassifier( criterion="entropy"),
 "logistic-regression":linear_model.LogisticRegression(C=0.01),
 "rf": ensemble.RandomForestClassifier(n_estimators=100)}
