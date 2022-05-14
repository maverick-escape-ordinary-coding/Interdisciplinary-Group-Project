# %%
# !/usr/bin/env python3
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import sys
import psycopg2
import pandas as pd
import numpy as np
import numbers

# visualisations
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import style
style.use('fivethirtyeight')
import seaborn as sns
import shap

# modelling helpers
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

from data_from_postgresql import postgresql_to_dataframe
from data_adjustment import adjust_data
from predict_salary_helper import help_norm_salary
from outliers_treatment import tukeys_method
from regressor_nn import create_model_fair, create_model_all, loss_history_model

from keras.initializers import RandomUniform
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.callbacks import EarlyStopping
from keras.layers import Dense


#%%
# connection to database
try:

    conn = psycopg2.connect(dbname='', user=, host='', port='',
                            password=)
    print('Connection successful')

    # those are my dataset columns, so requires changing to yours
    col_names = ['employee', 'division', 'groups', 'area', 'team', 'location', 'job_title', 'status', 'category',
                 'join_date', 'company_tenure',
                 'service_length', 'pay_amount', 'hourly_equivalent', 'notice_rule', 'ethnic_origin', 'nationality',
                 'gender', 'disabled', 'work_style',
                 'days_per_week', 'contract', 'appointment_type', 'marital_status', 'qualified_solicitor',
                 'benefit_level', 'full_time_equivalent',
                 'currency', 'ft_hours', 'ft_pay_amount', 'wp_description', 'career_effective_date',
                 'career_change_reason', 'pay_from_date',
                 'pay_change_reason', 'non_executive_director', 'payroll_name', 'tax', 'net_pay', 'basic_pay_nl_uk',
                 'payroll', 'post_name',
                 'arc_pension', 'auto_daily_rate', 'auto_hourly_rate', 'ees_nic', 'total_gross', 'city',
                 'salary_per_week_35ft', 'salary_per_month_35ft',
                 'salary_annual_35ft', 'salary_plus_bonus_35ft', 'role_level', 'career_level', 'hours_per_week',
                 'total_bonus_amount']

    client_a_df = postgresql_to_dataframe(conn, "select * from client_a", col_names)

except psycopg2.DatabaseError:
    sys.exit('Failed to connect to database')
finally:
    if conn is not None:
        conn.close()


#%%
'''
this function is very for my dataset - changing some columns to numerical, adding some categorical variables (like
ranges of salary, length of service etc.) 
I am attaching this function in other file so you can check and maybe make it more for yours, but if not just 
delete or comment  this line
'''
client_a_df = adjust_data(client_a_df)

#%% OUTLIERS
# finding outliers in salary and create new dataset without them to train model
prob_outliers, _ = tukeys_method(client_a_df, 'salary_per_month_35ft')

client_no_outliers = client_a_df.drop(prob_outliers)
client_no_outliers = client_a_df.fillna(0)


#%%
# choosing columns which, I think, can be only useful and enough (ALSO TO CHANGE WITH YOURS)
client_to_predict = client_no_outliers[['division', 'groups', 'area', 'team', 'location', 'job_title', 'status',
                                        'category', 'notice_rule', 'ethnic_origin', 'nationality', 'gender', 'disabled',
                                        'work_style', 'days_per_week', 'contract', 'appointment_type', 'marital_status',
                                        'qualified_solicitor', 'benefit_level', 'career_change_reason', 'payroll_name',
                                        'payroll', 'post_name', 'salary_per_month_35ft', 'role_level', 'career_level',
                                        'total_bonus_amount', 'service_length_years', 'service_length_category',
                                        'company_tenure_category', '35ft_equivalent_monthly_category', 'if_bonus']]



#%%
# changing categorical variables with string values to numerical labels (ALSO TO CHANGE WITH YOURS)
for item in ['division', 'groups', 'area', 'team', 'location', 'job_title', 'status', 'category',
             'notice_rule', 'ethnic_origin', 'nationality', 'gender', 'disabled', 'work_style',
             'contract', 'appointment_type', 'marital_status', 'qualified_solicitor',
             'benefit_level', 'career_change_reason', 'payroll_name', 'payroll', 'post_name',
             'service_length_category', 'company_tenure_category',
             '35ft_equivalent_monthly_category', 'if_bonus']:
    le = preprocessing.LabelEncoder()
    le.fit(client_to_predict[item])

    client_to_predict[item] = le.transform(client_to_predict[item])

    zip_iterator = zip(le.transform(le.classes_), le.classes_)
    item_dictionary = dict(zip_iterator)

    # creating dataframes with labels corresponding to value to save information about labels meaning
    globals()[f"df_{item[:4]}"] = pd.DataFrame.from_dict(item_dictionary, orient='index')

#%%
# choosing only variables which, I think, can only have impact on fair, unbiased salary (without salary info)
fair_client = client_to_predict[['division', 'groups', 'area', 'team', 'location', 'job_title', 'category',
                               'days_per_week', 'contract', 'appointment_type', 'qualified_solicitor',
                               'career_change_reason', 'payroll_name', 'payroll', 'post_name',
                               'role_level', 'career_level', 'service_length_years', 'service_length_category',
                               'company_tenure_category']]

# all variables, including biased (without salary)
# I included ethics, but may be deleted
all_features_client = client_to_predict[['division', 'groups', 'area', 'team', 'location', 'job_title', 'status',
                                       'category', 'notice_rule', 'ethnic_origin', 'nationality', 'gender', 'disabled',
                                       'work_style', 'days_per_week', 'contract', 'appointment_type', 'marital_status',
                                       'qualified_solicitor', 'benefit_level', 'career_change_reason', 'payroll_name',
                                       'payroll', 'post_name', 'role_level', 'career_level',
                                       'service_length_years', 'service_length_category', 'company_tenure_category',
                                       'if_bonus']]

salary_range = client_to_predict['35ft_equivalent_monthly_category']  # salary range - categories
actual_salary = client_to_predict['salary_per_month_35ft']    # actual salary - vector to learn and predict


#%%
# creating vectors for model inputs (dataframes to numpy arrays)
X_fair = fair_client.values
X_all = all_features_client.values
Y = actual_salary.values

#%%
# normalisation
scalar1 = MinMaxScaler()
scalar2 = MinMaxScaler()
scalar3 = MinMaxScaler()
scalar1.fit(X_fair)
scalar2.fit(X_all)
Y = Y.reshape(-1, 1)
scalar3.fit(Y)
X_fair = scalar1.transform(X_fair)
X_all = scalar2.transform(X_all)
Y = scalar3.transform(Y)

#%%
# building and training regressor model with cross validation for fair data
model_fair = create_model_fair()        # function in file regressor_nn.py - CHANGE IMPUT SIZE
estimator = KerasRegressor(build_fn=create_model_fair, epochs=70, batch_size=5, verbose=0)
estimator.fit(X_fair, Y)
kfold = KFold(n_splits=10)
results_fair = cross_val_score(estimator, X_fair, Y, cv=kfold)
loss_history_model(model_fair, X_fair, Y)   # plotting loss graph
# print("Baseline: %.2f (%.2f) MSE" % (results_fair.mean(), results_fair.std()))

#%%
# predincting fair salary for every employee
prediction_fair = estimator.predict(X_fair)
prediction_all = prediction_all.reshape(-1,1)
prediction_all = scalar3.inverse_transform(prediction_all)


#%%
# building and training regressor model with cross validation for all data
model_all = create_model_all()        # function in file regressor_nn.py - CHANGE IMPUT SIZE
estimator = KerasRegressor(build_fn=create_model_all, epochs=70, batch_size=5, verbose=0)
estimator.fit(X_all, Y)
kfold = KFold(n_splits=10)
results_all = cross_val_score(estimator, X_all, Y, cv=kfold)
loss_history_model(model_all, X_all, Y)   # plotting loss graph
# print("Baseline: %.2f (%.2f) MSE" % (results_all.mean(), results_all.std()))

#%%
# predincting salary for every employee considering all features
prediction_all = estimator.predict(X_all)


#%%
# making one dataframe from actual salary, predicted fairly and biased predicted
# to have a clear view and comparison of all employees
predicts = pd.DataFrame({'fair_predictions' : prediction_fair, 'all_features_predictions' : prediction_all})
salary_with_predicts = pd.concat([actual_salary, predicts], axis=1)

#%%
pay_labels = []
for _, case in salary_with_predicts.iterrows():
    if case.fair_predictions > case.salary_per_month_35ft:
        pay_labels.append('underpaid')
    else:
        pay_labels.append('overpaid')

# adding to dataframe above information if predicted salary is smaller or bigger than actual and info about gender
salary_with_predicts['pay_label'] = pay_labels
salary_with_predicts['gender'] = client_a_df['gender']

#%%
# plotting boxplot with over- and underpaid employees per gender
plt.figure(figsize=(15, 15))
sns.boxplot(data = salary_with_predicts, x=pay_labels, y='salary_per_month_35ft', hue='gender', palette='husl',
            showfliers=False)
plt.xticks(rotation=-70)
plt.ylabel("Monthly Income FTE", fontsize=12)
plt.xlabel(item, fontsize=12)
plt.title("Illustration of the Wage for Different pay label per gender", fontsize=15)
plt.savefig('boxplot_predict_label_gender.png', bbox_inches='tight')
plt.show()

#%%
# plotting histogram with over- and underpaid employees per gender
plt.figure(figsize=(15, 15))
sns.catplot(data = salary_with_predicts, x=pay_labels, hue='gender', kind="count", palette='husl')
plt.xticks(rotation=-70)
plt.ylabel("Monthly Income FTE", fontsize=12)
plt.xlabel(item, fontsize=12)
plt.title("Illustration of the Wage for Different pay label per gender", fontsize=15)
plt.savefig('boxplot_predict_label_gender.png', bbox_inches='tight')
plt.show()

#%%
shap.initjs()

X_train_summary = shap.kmeans(X_fair, 40)
explainer = shap.KernelExplainer(model_fair, X_train_summary)
shap_values = explainer.shap_values(X_fair)

#%%
X_train_sample = fair_client.sample(400)
shap_values  = explainer.shap_values(X_train_sample)

#%%
shap.summary_plot(shap_values, X_train_sample, show=False)
plt.savefig("fair_feature_importance.png", bbox_inches='tight')
plt.close()


#%%

shap.initjs()

X_train_summary = shap.kmeans(X_all, 40)
explainer = shap.KernelExplainer(model_all, X_train_summary)
shap_values = explainer.shap_values(X_all)

#%%
X_train_sample = all_features_client.sample(400)
shap_values = explainer.shap_values(X_train_sample)

#%%
shap.summary_plot(shap_values, X_train_sample, show=False)
plt.savefig("all_feature_importance.png", bbox_inches='tight')
plt.close()
