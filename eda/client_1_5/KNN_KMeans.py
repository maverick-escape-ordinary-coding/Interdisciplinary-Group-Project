import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
%matplotlib inline

data_model=pd.read_csv('client_1_5_as_model.csv')
data_model

print (data_model.dtypes)

#Split into train/test
X = np.array(data_model.select_dtypes(include='number'))
y = np.array(data_model['pay_label']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3) #p=2 is euclidean default
knn.fit (X_train, y_train)    
        
print('KNN Score for the Training Data :',knn.score(X_train, y_train)) 
print('KNN Score for the Ttesting Data :',knn.score(X_test, y_test))


training_accuracy =[]
test_accuracy =[]
#try n neighbours 1 to 10
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    #build the model
    knn2 = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn2.fit (X_train, y_train)
    training_accuracy.append(knn2.score(X_train, y_train))
    test_accuracy.append(knn2.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

#Calculating F1 score
Pa = 229/(229+12+20)
Ra = 229/(229+4+11)

Pb = 68/(68+4+0)
Rb = 68/(68+12+0)

Pc = 217/(217+0+11)
Rc = 217/(217+0+20)

P = ((Pa*(229+4+11))+(Pb*(68+12+0))+(Pc*(217+0+20)))/(220+4+11+12+68+20+217)
R = ((Ra*(229+4+11))+(Rb*(68+12+0))+(Rc*(217+0+20)))/(220+4+11+12+68+20+217)

F1_Score = (2*P*R)/(P+R)
F1_Score


from sklearn.model_selection import cross_val_score
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)

#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

#print each cv score (accuracy) and average them
print('CV Scores from Training the Model :',cv_scores)
print('cv_scores mean: {}'.format(np.mean(cv_scores)))


from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

#fit model to data
knn_gscv.fit(X, y)

#check top performing n_neighbors value
print('Estimated :',knn_gscv.best_params_)
print('Estimated test score :',knn_gscv.best_score_)


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

X2 = np.array(data_model.select_dtypes(include='number'))
y2 = np.array(data_model['pay_label']) 
data_model['pay_label'] = data_model['pay_label'].replace("underpaid", 0)
data_model['pay_label'] = data_model['pay_label'].replace("fairly_paid", 1)
data_model['pay_label'] = data_model['pay_label'].replace("overpaid", 2)

df = data_model.select_dtypes(include='number')
del df['employee']



#Look for elbow to determine clusters
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X2).score(X2) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


from sklearn.preprocessing import MinMaxScaler
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

from sklearn.metrics import silhouette_score
KMean= KMeans(n_clusters=2)
KMean.fit(df)
label=KMean.predict(df)
#Calculating the silhouette score:
print(f'Silhouette Score(n=2): {silhouette_score(df, label)}')  
print()
print('Closer to 1, the further apart the clusters')

kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_

plt.scatter(df['quasi_predicted_salary'], df['hourly_salary_plus_fte'],c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel('quasi_predicted_salary')
plt.ylabel('hourly_salary_plus_fte')
plt.title('Observing KMeans Scattering')
plt.show()
