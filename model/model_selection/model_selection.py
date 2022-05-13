#%%
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, SGDRegressor
import matplotlib.pyplot as plt
import sys
sys.path.append("../../data_helper")
from load_transform import load_transform, get_config

# a list of models to evaluate
def get_models():
    models = dict()
    models['BR'] = BayesianRidge()
    #models['EN'] = ElasticNet()
    models['SGD'] = SGDRegressor()
    models['GBR'] = GradientBoostingRegressor()
    #models['LR'] = LinearRegression()
    models['RFR'] = RandomForestRegressor(n_estimators=50)
    return models
 
# cross-validation
def evaluate_model(model, X, y):
    #cv = LeaveOneOut()
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
    scores = cross_val_score(model, X, y, scoring='r2', cv=cv, error_score='raise')
    return scores

#takes the first arg on the command line as the config filename
if __name__ == "__main__":
    if (len(sys.argv) > 1):
        config_file_name = sys.argv[1]
    else:
        config_file_name = None

X, y, X_full, y_full, client_df = load_transform(config_file_name)

#%%
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

