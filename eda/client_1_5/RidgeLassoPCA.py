#Using AS method to convert client 1.5 to some sort of numerical value

import pandas as pd

df=pd.read_csv("client_1_5_as_model.csv")
df['pay_label'].value_counts()

df['gender'].replace('Female',2,inplace=True)
df['gender'].replace('Male',1,inplace=True)
df

from sklearn.preprocessing import StandardScaler

features = df[['hourly_salary','full_time_equivalent','annual_salary', '**numeric_pay_label',
 'bonus_non_voucher','total_bonus', 'bonus_tf','gender','norm_salary_location','norm_salary_division',
 'norm_salary_post_name','norm_salary_job_level', 'norm_salary_job_group','total_package','norm_salary_category',
 'age','norm_salary_age','years_service','norm_salary_years_service']]

# Separating out the features
X = df.iloc[:, [0,1,2,8,9,10,11,13,15,17,19,21,25,27,28,30]].values
y = df.iloc[:, 6].values

X = StandardScaler().fit_transform(X)

X_columns = df.iloc[:, [0,1,2,8,9,10,11,13,15,17,19,21,25,27,28,30]].columns.values

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf

finalDf = pd.concat([principalDf, df[['pay_label']]], axis = 1)
#finalDf.drop(2,0,inplace=True)
finalDf

import matplotlib.pyplot as plt

#Plot with all the data
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
pay_labels = ['underpaid', 'fairly_paid', 'overpaid']
colors = ['r', 'g', 'b']

for pay_label, color in zip(pay_labels,colors):
    indicesToKeep = finalDf['pay_label'] == pay_label
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(pay_labels)
ax.grid()

#Plot taking out a couple of outliers 
finalDf = pd.concat([principalDf, df[['pay_label']]], axis = 1)
finalDf.drop(2,0,inplace=True)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
pay_labels = ['underpaid', 'fairly_paid', 'overpaid']
colors = ['r', 'g', 'b']

for pay_label, color in zip(pay_labels,colors):
    indicesToKeep = finalDf['pay_label'] == pay_label
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(pay_labels)
ax.grid()


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split( 
    df.iloc[:, [0,1,2,8,9,10,11,13,15,17,19,21,25,27,28,30]].values,
    df.iloc[:, 6].values, test_size=1/7.0, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

from sklearn.decomposition import PCA
pca = PCA(.95)
pca.fit(train_X)

train_X = pca.transform(train_X)
test_X = pca.transform(test_X)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

X2 = df.iloc[:, [0,1,2,8,9,10,11,13,15,17,19,21,25,27,28,30]].values
y2 = df.iloc[:, 6].values
X2 = StandardScaler().fit_transform(X)
X2_columns = df.iloc[:, [0,1,2,8,9,10,11,13,15,17,19,21,25,27,28,30]].columns.values

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X2, y2)
forest = RandomForestClassifier(n_estimators=10)
forest.fit(X2, y2)

def plot_feature_importance(model):
    n_features = X2.shape[1]
    plt.barh(range(n_features), model.feature_importances_,align='center' )
    plt.yticks(np.arange(n_features), X2_columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importance(forest)

from sklearn.linear_model import Ridge, Lasso
X3 = df.iloc[:, [0,1,2,8,9,10,11,13,15,17,19,21,25,27,28,30]].values
y3 = df.iloc[:, 7].values
X3 = StandardScaler().fit_transform(X)

import itertools

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=0)

#%% lasso --~O
markers = itertools.cycle(('+', 'o', '*', 'X', 'd')) 
alphas = [1, 0.1, 0.01, 0.0001]
for alpha in alphas:  
    lasso = Lasso(alpha=alpha, max_iter=10e5)
    lasso.fit(X3_train,y3_train)
    train_score=lasso.score(X3_train,y3_train)
    test_score=lasso.score(X3_test,y3_test)
    coeff_used = np.sum(lasso.coef_!=0)
    print("Lasso train/test scores for alpha={:.2f} {:.2f}/{:.2f}".format(alpha, train_score, test_score))
    print("Coefs used={:.2f}".format(coeff_used))
    plt.plot(lasso.coef_, marker=next(markers), alpha = 0.7, linestyle='none',label=r'alpha = {}'.format(alpha))
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=2)
plt.show

#%% ridge --^^--
markers = itertools.cycle(('+', 'o', '*', 'X', 'd')) 
alphas = [0.01, 1, 10, 100]

for alpha in alphas:    
    ridge = Ridge(alpha=alpha)
    ridge.fit(X3_train, y3_train)
    train_score = ridge.score(X3_train,y3_train)
    test_score = ridge.score(X3_test, y3_test)
    print("Ridge train/test scores for alpha={:.2f} {:.2f}/{:.2f}".format(alpha, train_score, test_score))
    plt.plot(ridge.coef_, marker=next(markers), linestyle='none',label=r'alpha = {}'.format(alpha))

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=1)
plt.show()

