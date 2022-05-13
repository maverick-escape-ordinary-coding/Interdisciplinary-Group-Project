#%% setup / create Xy
import pandas as pd
import numpy as np
import psycopg2
import configparser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import itertools

config = configparser.ConfigParser()
config.read('app_client_1_5.ini')
db_params = config['DB']
app_params = config['APP']

#load data from SQL
con = psycopg2.connect(host='localhost', database=db_params['db'], user = db_params['user'], password = db_params['password'] )
sql = "SELECT * from %s" % (db_params['table'])
client_df = pd.read_sql_query(sql, con)


if ('weekly_hours_column' in app_params):
    client_df['_annual_salary_fte'] = client_df[app_params['actual_salary_column']] * (int(app_params['fte_hours']) / client_df[app_params['weekly_hours_column']])
else:
    client_df['_annual_salary_fte'] = client_df[app_params['actual_salary_column']]

client_df['_total_annual_pkg'] = client_df[app_params['actual_salary_column']]

if ('benefits_column_list' in app_params):
    benefit_colmuns = app_params['benefits_column_list'].split(',')
    benefit_colmuns = [s.strip() for s in benefit_colmuns]
    for benefit_column in benefit_colmuns:
        #hax!, db contains stringy floats
        try:
            client_df[benefit_column].apply(float)
            client_df['_total_annual_pkg'] += client_df[benefit_column]
        except:
            print('Float conversion failed')
            
if ('weekly_hours_column' in app_params):
    client_df['_total_annual_pkg_fte'] = client_df['_total_annual_pkg'] * (int(app_params['fte_hours']) / client_df[app_params['weekly_hours_column']])
else: 
    client_df['_total_annual_pkg_fte'] = client_df['_total_annual_pkg']
    
#treat outliers and scale total package if needed
if ('outlier_percentiles' in app_params):
    values = [s.strip() for s in app_params['outlier_percentiles'].split(',')]
    low, high = client_df['_total_annual_pkg_fte'].quantile( [float(values[0]), float(values[1])] )
    client_df = client_df.query("_total_annual_pkg_fte >= {} & _total_annual_pkg_fte <= {}".format(low, high))
    #reset index, make it linear again for ease of joining below
    client_df = client_df.reset_index()
    
scaler = MinMaxScaler(feature_range=(0, 1))
#note this scales directly on the original dataframe, use only for feature selection.
#set to false when predicting.
if ('scale_target' in app_params and app_params['scale_target'].lower() == 'true'):
    client_df['_total_annual_pkg_fte'] = scaler.fit_transform(client_df['_total_annual_pkg_fte'].values.reshape(-1, 1))

#oh, cat
df_oh_cat_features = df_label_cat_features = df_num_features = None

#produce a dataframe that contains our one hot encoded features
if ('oh_cat_feature_list' in app_params):
    oh_cat_features = app_params['oh_cat_feature_list'].split(',')
    oh_cat_features = [s.strip() for s in oh_cat_features]
    df_oh_cat_features = pd.get_dummies(data=client_df[oh_cat_features], drop_first=True)
    if ('scale_cat_features' in app_params and app_params['scale_cat_features'].lower() == 'true'):
        for column in df_oh_cat_features.columns:
            df_oh_cat_features[column] = scaler.fit_transform(df_oh_cat_features[column].values.reshape(-1, 1)) 

#produce a dataframe that contains our label encoded features
if ('label_cat_feature_list' in app_params):
    label_cat_features = app_params['label_cat_feature_list'].split(',')
    label_cat_features = [s.strip() for s in label_cat_features]
    le = LabelEncoder()
    for feature in label_cat_features:
        if (type(df_label_cat_features) != pd.DataFrame):
            df_label_cat_features = pd.DataFrame()
        le.fit(client_df[feature])
        df_label_cat_features[feature] = le.transform(client_df[feature])
        if ('scale_label_features' in app_params and app_params['scale_label_features'].lower() == 'true'):
             df_label_cat_features[feature] = scaler.fit_transform(df_label_cat_features[feature].values.reshape(-1, 1)) 
                

#produce a dataframe that contains our numerical features
if ('num_feature_list' in app_params):
    num_features = app_params['num_feature_list'].split(',')
    num_features = [s.strip() for s in num_features]
    df_num_features = client_df[num_features]
    if ('scale_num_features' in app_params and app_params['scale_num_features'].lower() == 'true'):
        for feature in num_features:
            df_num_features[feature] = scaler.fit_transform(df_num_features[feature].values.reshape(-1, 1))

#create X from our three feature dataframes if they have been initialised
X = pd.DataFrame()
for features in (df_oh_cat_features, df_num_features, df_label_cat_features): #
    if type(features) == pd.DataFrame:
        if (X.empty):
            X = features
        else:    
            X = X.join(features)
            
drop = False
if ('train_split_column' in app_params):
    if (app_params['train_split_column'] not in X):
        drop = True
        X[app_params['train_split_column']] = client_df[app_params['train_split_column']]
    X = X.query("{} == '{}'".format(app_params['train_split_column'], app_params['train_split_value']))
    y = client_df.query("{} == '{}'".format(app_params['train_split_column'], app_params['train_split_value']))['total_annual_pkg_fte']
    if drop:
        X.drop(app_params['train_split_column'], axis = 1, inplace=True )
else:
    y = client_df['_total_annual_pkg_fte']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#%% lasso --~O
markers = itertools.cycle(('+', 'o', '*', 'X', 'd')) 
alphas = [1, 0.1, 0.01, 0.0001]

for alpha in alphas:  
    lasso = Lasso(alpha=alpha, max_iter=10e5)
    lasso.fit(X_train,y_train)
    train_score=lasso.score(X_train,y_train)
    test_score=lasso.score(X_test,y_test)
    coeff_used = np.sum(lasso.coef_!=0)
    print("Lasso train/test scores for alpha={} {:.2f}/{:.2f}".format(alpha, train_score, test_score))
    print("Coefs used={:.2f}".format(coeff_used))
    plt.plot(lasso.coef_, marker=next(markers), alpha = 0.7, linestyle='none',label=r'alpha = {}'.format(alpha))
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=2)
plt.show()

#%% ridge --^^--
markers = itertools.cycle(('+', 'o', '*', 'X', 'd')) 
alphas = [0.01, 1, 10, 100]

for alpha in alphas:    
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    train_score = ridge.score(X_train,y_train)
    test_score = ridge.score(X_test, y_test)
    print("Ridge train/test scores for alpha={:.2f} {:.2f}/{:.2f}".format(alpha, train_score, test_score))
    plt.plot(ridge.coef_, marker=next(markers), linestyle='none',label=r'alpha = {}'.format(alpha))

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=2)
plt.show()

#%%PCA

#create some targets
target_ranges = [(10,39), (40,69), (70,99), (100,129), (130,159), (160,189), (190,219), (220,9999)]
targets = set() 
def get_pkg_range(package):
    thou = (int((package)/10000))*10
    for target_range in target_ranges:
        if (thou >= target_range[0] and thou <= target_range[1]):
            range = str(target_range[0]) + " - " + str(target_range[1])
            targets.add(range)
            return (range)

client_df['_total_annual_pkg_fte_range'] = client_df['_total_annual_pkg_fte'].apply(get_pkg_range)

n = 2

pca = PCA(n_components=n)

columns=[]
for column_no in range(1,n+1):
    columns.append('PC {}'.format(column_no))

pc = pca.fit_transform(X)

#merge the targets back in
final_df = pd.DataFrame(data = pc, columns = columns)

final_df = pd.concat([final_df, client_df[['_total_annual_pkg_fte_range']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
#colors = ['r', 'g', 'b']
for target in targets:
    indicesToKeep = final_df['_total_annual_pkg_fte_range'] == target
    ax.scatter(final_df.loc[indicesToKeep, 'PC 1']
               , final_df.loc[indicesToKeep, 'PC 2']        
               , s = 50)
ax.legend(targets)
ax.grid()

# %%



