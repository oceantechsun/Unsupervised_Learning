#dim redux and rerun on cluster
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import time
from sklearn.model_selection import cross_val_score

import pandas as pd
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPClassifier


#wine = load_wine()
url = "/Users/jordan/Desktop/Machine Learning/Unsupervised Learning/winequality-white.csv"

# Let's start by naming the features
names = ["fixed acidity", "volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",\
    "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

dataset = pd.read_csv(url, names=names, sep=';').iloc[1:,:]

X = dataset.iloc[:,:-1]
# Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
scaler = StandardScaler()
scaler.fit(X_train)

X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

from sklearn.decomposition import PCA

#principalComponents = pca.fit_transform(X_train)
#principalDf = pd.DataFrame(data = principalComponents)

homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []
acc_scores = []
cross_vals_1 = []
cross_vals_2 = []
cross_vals_3 = []
times = []
for c in range(3, X_train.shape[1]+1):
    num_components.append(c)
    #print(c)
    #print('hi')
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    pca = PCA(n_components=c)
    #pca = pca.fit(X_train)

    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)
    start = time.time()

    neuron = MLPClassifier(hidden_layer_sizes=(1+c),max_iter=100)

    neuron.fit(X_train, y_train)
    cross_vals_1.append(cross_val_score(neuron, X_test, y_test, cv = 3)[0])
    cross_vals_2.append(cross_val_score(neuron, X_test, y_test, cv = 3)[1])
    cross_vals_3.append(cross_val_score(neuron, X_test, y_test, cv = 3)[2])
    print(cross_val_score(neuron, X_test, y_test, cv = 3))
    end = time.time()

    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    #finalDf = clean_dataset(finalDf)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)
    #print(X_train_next)
    km = KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = km.predict(X_test_next)
    end = time.time()
    times.append(end-start)
    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)



print('cross vals')
#print(cross_vals)
print('times')
print(times)

    #acc_scores.append(accuracy_score(y_test, y_pred))

"""
[0.54471545 0.52342159 0.53798768]
[0.55487805 0.53360489 0.5174538 ]
[0.54674797 0.52138493 0.50718686]
[0.54065041 0.51323829 0.52156057]
"""
fig3, ax3 = plt.subplots()
ax3.plot(num_components, cross_vals_1)
ax3.plot(num_components, cross_vals_2)
ax3.plot(num_components, cross_vals_3)
#plt.legend(['without cluster label', 'with EM label', 'with Kmeans label'])
plt.xlabel("Number of components")
plt.ylabel("Cross Validation score")
plt.title("Neural network accuracy")
plt.savefig('pca_cv.png')



X = dataset.iloc[:,:-1]
# Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
scaler = StandardScaler()
scaler.fit(X_train)

X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)


#cross_vals = []
cross_vals_1 = []
cross_vals_2 = []
cross_vals_3 = []
times = []
for c in range(3, X_train.shape[1]+1):
    #print(c)
    #print('hi')
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    pca = FastICA(n_components=c)
    #pca = pca.fit(X_train)

    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)
    start = time.time()

    neuron = MLPClassifier(hidden_layer_sizes=(1+c),max_iter=100)

    neuron.fit(X_train, y_train)
    cross_vals_1.append(cross_val_score(neuron, X_test, y_test, cv = 3)[0])
    cross_vals_2.append(cross_val_score(neuron, X_test, y_test, cv = 3)[1])
    cross_vals_3.append(cross_val_score(neuron, X_test, y_test, cv = 3)[2])
    #cross_vals.append(cross_val_score(neuron, X_test, y_test, cv = 3))
    print(cross_val_score(neuron, X_test, y_test, cv = 3))
    

    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    #finalDf = clean_dataset(finalDf)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)
    #print(X_train_next)
    km = KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = km.predict(X_test_next)
    end = time.time()
    times.append(end-start)
    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))
    
    #print( acc_scores)



print('cross vals')
print(cross_vals_2)
print('times')
print(times)

    #acc_scores.append(accuracy_score(y_test, y_pred))

"""
[0.54471545 0.52342159 0.53798768]
[0.55487805 0.53360489 0.5174538 ]
[0.54674797 0.52138493 0.50718686]
[0.54065041 0.51323829 0.52156057]
"""

fig3, ax3 = plt.subplots()
ax3.plot(num_components, cross_vals_1)
ax3.plot(num_components, cross_vals_2)
ax3.plot(num_components, cross_vals_3)
#plt.legend(['without cluster label', 'with EM label', 'with Kmeans label'])
plt.xlabel("Number of components")
plt.ylabel("Cross Validation score")
plt.title("Neural network accuracy")
plt.savefig('ica_cv.png')

X = dataset.iloc[:,:-1]
# Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
scaler = StandardScaler()
scaler.fit(X_train)

X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

cross_vals_1 = []
cross_vals_2 = []
cross_vals_3 = []
times = []
for c in range(3, X_train.shape[1]+1):
    #print(c)
    #print('hi')
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    pca = random_projection.GaussianRandomProjection(n_components=c)
    #pca = pca.fit(X_train)

    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)
    start = time.time()

    neuron = MLPClassifier(hidden_layer_sizes=(1+c),max_iter=100)

    neuron.fit(X_train, y_train)
    cross_vals_1.append(cross_val_score(neuron, X_test, y_test, cv = 3)[0])
    cross_vals_2.append(cross_val_score(neuron, X_test, y_test, cv = 3)[1])
    cross_vals_3.append(cross_val_score(neuron, X_test, y_test, cv = 3)[2])
    #cross_vals.append(cross_val_score(neuron, X_test, y_test, cv = 3))
    print(cross_val_score(neuron, X_test, y_test, cv = 3))
    

    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    #finalDf = clean_dataset(finalDf)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)
    #print(X_train_next)
    km = KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = km.predict(X_test_next)
    end = time.time()
    times.append(end-start)
    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))
    
    #print( acc_scores)



print('cross vals_')
print(cross_vals_1)
print('times')
print(times)

    #acc_scores.append(accuracy_score(y_test, y_pred))

"""
[0.54471545 0.52342159 0.53798768]
[0.55487805 0.53360489 0.5174538 ]
[0.54674797 0.52138493 0.50718686]
[0.54065041 0.51323829 0.52156057]
"""

fig3, ax3 = plt.subplots()
ax3.plot(num_components, cross_vals_1)
ax3.plot(num_components, cross_vals_2)
ax3.plot(num_components, cross_vals_3)
#plt.legend(['without cluster label', 'with EM label', 'with Kmeans label'])
plt.xlabel("Number of components")
plt.ylabel("Cross Validation score")
plt.title("Neural network accuracy")
plt.savefig('rand_cv.png')


X = dataset.iloc[:,:-1]
# Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
scaler = StandardScaler()
scaler.fit(X_train)

X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)


#cross_vals = []
cross_vals_1 = []
cross_vals_2 = []
cross_vals_3 = []
times = []
for c in range(3, X_train.shape[1]+1):
    #print(c)
    #print('hi')
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    pca = LinearDiscriminantAnalysis(n_components = c)
    #pca = pca.fit(X_train)

    principalComponents = pca.fit_transform(X, y)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)
    start = time.time()

    neuron = MLPClassifier(hidden_layer_sizes=(1+c),max_iter=100)

    neuron.fit(X_train, y_train)
    cross_vals_1.append(cross_val_score(neuron, X_test, y_test, cv = 3)[0])
    cross_vals_2.append(cross_val_score(neuron, X_test, y_test, cv = 3)[1])
    cross_vals_3.append(cross_val_score(neuron, X_test, y_test, cv = 3)[2])
    #cross_vals.append(cross_val_score(neuron, X_test, y_test, cv = 3))
    print(cross_val_score(neuron, X_test, y_test, cv = 3))
    

    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    #finalDf = clean_dataset(finalDf)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)
    #print(X_train_next)
    km = KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = km.predict(X_test_next)
    end = time.time()
    times.append(end-start)
    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))
    
    #print( acc_scores)



print('cross vals')
print(cross_vals_3)
print('times')
print(times)

    #acc_scores.append(accuracy_score(y_test, y_pred))

"""
[0.54471545 0.52342159 0.53798768]
[0.55487805 0.53360489 0.5174538 ]
[0.54674797 0.52138493 0.50718686]
[0.54065041 0.51323829 0.52156057]
"""

fig3, ax3 = plt.subplots()
ax3.plot(num_components, cross_vals_1)
ax3.plot(num_components, cross_vals_2)
ax3.plot(num_components, cross_vals_3)
#plt.legend(['without cluster label', 'with EM label', 'with Kmeans label'])
plt.xlabel("Number of components")
plt.ylabel("Cross Validation score")
plt.title("Neural network accuracy")
plt.savefig('line_cv.png')