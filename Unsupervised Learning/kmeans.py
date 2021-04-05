from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
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



homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []

for c in range(3,12):
    gmm = GaussianMixture(n_components=c, random_state=2).fit(X_train)

    #kmeans.fit(X_train)
    #pca = PCA(n_components=7)
    #pca.fit(X_features)
    #x_3d = pca.transform(X_features)

    pred_y_test = gmm.predict(X_test)

    comp_score = metrics.completeness_score(y_test, pred_y_test)
    homo_score = metrics.homogeneity_score(y_test, pred_y_test)
    var_score = gmm.score(X_test)
    sil_score = metrics.silhouette_score(X_test, pred_y_test, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    aic_vals.append(gmm.aic(X_test))
    bic_vals.append(gmm.bic(X_test))
    rmse_vals.append(mean_squared_error(y_test, pred_y_test))

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('Number of clusters')
plt.title('Performance evaluation scores for EM')
plt.grid(True)

plt.savefig('wine_test4.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster for EM')
plt.xlabel('Number of cluster')
plt.grid(True)

plt.savefig('wine_test5.png')


fig5, ax5 = plt.subplots()
ax5.plot(num_components, aic_vals)
ax5.plot(num_components, bic_vals)
plt.title('AIC/BIC values for EM')
plt.xlabel('Number of cluster')
plt.grid(True)

plt.savefig('wine_test6.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error fo rEM')
plt.xlabel('Number of cluster')
plt.grid(True)

plt.savefig('wine_test7.png')



homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
rmse_vals = []


for c in range(3,12):
    kmeans = KMeans(n_clusters=c, random_state=0).fit(X_train)

    kmeans.fit(X_train)
    #pca = PCA(n_components=7)
    #pca.fit(X_features)
    #x_3d = pca.transform(X_features)

    pred_y_test = kmeans.predict(X_test)

    comp_score = metrics.completeness_score(y_test, pred_y_test)
    homo_score = metrics.homogeneity_score(y_test, pred_y_test)
    var_score = kmeans.score(X_test)
    sil_score = metrics.silhouette_score(X_test, pred_y_test, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    rmse_vals.append(mean_squared_error(y_test, pred_y_test))


    #plt.figure(figsize=(8,6))
    #plt.scatter(x_3d[:,1], x_3d[:,2], c=dataset['quality'])
    #plt.savefig('test.png')'

print(num_components)

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('Number of clusters')
plt.title('Performance evaluation scores for KMeans')
plt.grid(True)

plt.savefig('wine_test1.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance by cluster count for KMeans')
plt.xlabel('Number of clusters')
plt.grid(True)

plt.savefig('wine_test2.png')

plot, axis = plt.subplots()
axis.plot(num_components, rmse_vals)
plt.title('Mean Squared Error for KMeans')
plt.xlabel('Number of cluster')
plt.grid(True)

plt.savefig('wine_test0.png')

#mean_squared = []

#dim redux
#for c in range(3,12):
#    dim_frame = PCA(n_components = c, random_state = 2)
#    X_PCA = dim_frame.fit_transform(X)
#mean_squared.append()


pca = PCA(random_state=2).fit(X_train) #for all components
cum_var = np.cumsum(pca.explained_variance_ratio_)

fig, ax1 = plt.subplots()
ax1.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
ax1.set_xlabel('Principal Components')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Cumulative Explained Variance Ratio', color='b')
ax1.tick_params('y', colors='b')
plt.grid(False)

ax2 = ax1.twinx()
ax2.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'm-')
ax2.set_ylabel('Eigenvalues', color='m')
ax2.tick_params('y', colors='m')
plt.grid(False)

#plt.title("PCA Explained Variance and Eigenvalues: "+ title)
fig.tight_layout()
plt.savefig('wine_test3.png')




#reconstruction error for pca

recon_vals = []
recon_vals2 = [] 
for c in range (3,12):
    pca = PCA(n_components=c)#random_projection.GaussianRandomProjection(n_components = c)
    x1 = pca.fit_transform(X_train)
    x2 = x1.dot(pca.components_) + np.mean(X_train, axis = 0)
    x3 = X_train-x2
    x3 = np.mean(x3**2)
    

    recon_vals.append(x3)

fig2,ax2 = plt.subplots()
ax2.plot(num_components, recon_vals)
plt.xlabel('n_components')
plt.title('Reconstruction Error')
ax2.set_ylabel('Reconstruction Error')
plt.savefig('recon error for pca.png')
    



ica = FastICA(random_state=2).fit(X_train)
temp = ica.components_


transition_data = ica.transform(X_train)

kval = scipy.stats.kurtosis(transition_data, fisher=False)

finalk = sorted(kval, reverse = True)	
fig, ax = plt.subplots()

ax.bar(np.arange(X.shape[1]), finalk , linewidth=2, color = 'blue')

plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('kurtosis')
plt.savefig('con_test3b.png')


recon_vals = []
recon_vals2 = [] 
for c in range (3,12):
    rand = random_projection.GaussianRandomProjection(n_components = c)
    x1 = rand.fit_transform(X_train)
    x2 = x1.dot(rand.components_) + np.mean(X_train, axis = 0)
    x3 = X_train-x2
    x3 = np.mean(x3**2)
    

    recon_vals.append(x3)

fig2,ax2 = plt.subplots()
ax2.plot(num_components, recon_vals)
plt.axis('tight')
plt.xlabel('n_components')
plt.title('Reconstruction Error')

ax.set_ylabel('Reconstruction Error')
plt.savefig('recon error rand.png')

for c in range (3,12):
    temp_vals = []
    for b in range(5):
        rand = random_projection.GaussianRandomProjection(n_components = c)
        x1 = rand.fit_transform(X_train)
        x2 = x1.dot(rand.components_) + np.mean(X_train, axis = 0)
        x3 = X_train-x2
        x3 = np.mean(x3**2)
        temp_vals.append(x3)
    

    recon_vals2.append(np.std(temp_vals))
    #temp_vals.append(recon_vals2)
#standard_dev_recon = np.std(temp_vals, axis=0)

fig2,ax2 = plt.subplots()
ax2.plot(num_components, recon_vals2)
plt.axis('tight')
plt.xlabel('number of components')
plt.title('Variation in Reconstruction Error')
ax.set_ylabel('Standard Deviation of Reconstruction Error')
plt.savefig('Variation in recon error rand.png')  #std of recon error for rp











#dim redux and rerun on cluster



def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

from sklearn.decomposition import PCA

principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents)

homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []
acc_scores = []
for c in range(3, X_train.shape[1]+1):
    print(c)
    print('hi')
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    pca = PCA(n_components=c)
    #pca = pca.fit(X_train)

    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)


    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    #finalDf = clean_dataset(finalDf)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)
    print(X_train_next)
    km = KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = km.predict(X_test_next)

    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)


    comp_score = metrics.completeness_score(y_test_next, pred_y_test_next)
    homo_score = metrics.homogeneity_score(y_test_next, pred_y_test_next)
    var_score = km.score(X_test_next)
    sil_score = metrics.silhouette_score(X_test_next, pred_y_test_next, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    #aic_vals.append(km.aic(X_test_next))
    #bic_vals.append(km.bic(X_test_next))
    rmse_vals.append(mean_squared_error(y_test_next, pred_y_test_next))


    #acc_scores.append(accuracy_score(y_test, y_pred))

fig2,ax2 = plt.subplots()
ax2.plot(num_components, acc_scores)
plt.axis('tight')
plt.xlabel('components')
ax.set_ylabel('accuracy_score')
plt.savefig('kmeans and pca acc.png')  #std of recon error for rp

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('components')
plt.title('Performance evaluation scores')
plt.grid(True)
plt.savefig('perf km and pca.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster')
plt.xlabel('components')
plt.grid(True)
plt.savefig('var km pca.png')


fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error ')
plt.xlabel('components')
plt.grid(True)
plt.savefig('rmse km pca.png')






homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []

acc_scores = []
for c in range(3, X_train.shape[1]+1):
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    pca = PCA(n_components=c)
    #pca = pca.fit(X_train)

    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)


    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)

    em = GaussianMixture(n_components=c, random_state=2).fit(X_train_next)#KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = em.predict(X_test_next)

    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)

    comp_score = metrics.completeness_score(y_test_next, pred_y_test_next)
    homo_score = metrics.homogeneity_score(y_test_next, pred_y_test_next)
    var_score = em.score(X_test_next)
    sil_score = metrics.silhouette_score(X_test_next, pred_y_test_next, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    aic_vals.append(em.aic(X_test_next))
    bic_vals.append(em.bic(X_test_next))
    rmse_vals.append(mean_squared_error(y_test_next, pred_y_test_next))




    #acc_scores.append(accuracy_score(y_test, y_pred))
    
fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('components')
plt.title('Performance evaluation scores')
plt.grid(True)
plt.savefig('perf em and pca.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster')
plt.xlabel('components')
plt.grid(True)
plt.savefig('var em and pca.png')


fig5, ax5 = plt.subplots()
ax5.plot(num_components, aic_vals)
ax5.plot(num_components, bic_vals)
plt.title('AIC/BIC values')
plt.xlabel('components')
plt.grid(True)
plt.savefig('aic em and pca.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error')
plt.xlabel('components')
plt.grid(True)
plt.savefig('rmse em and pca.png')

fig2,ax2 = plt.subplots()
ax2.plot(num_components, acc_scores)
plt.axis('tight')
plt.xlabel('components')
ax.set_ylabel('accuracy score')
plt.savefig('em and pca acc.png')  #std of recon error for rp


homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []
acc_scores = []
for c in range(3, X_train.shape[1]+1):
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    ica = FastICA(n_components = c, random_state=2)
    #pca = pca.fit(X_train)

    principalComponents = ica.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)


    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)

    km = KMeans(n_clusters=c, random_state=2).fit(X_train_next)#KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = km.predict(X_test_next)

    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)

    comp_score = metrics.completeness_score(y_test_next, pred_y_test_next)
    homo_score = metrics.homogeneity_score(y_test_next, pred_y_test_next)
    var_score = km.score(X_test_next)
    sil_score = metrics.silhouette_score(X_test_next, pred_y_test_next, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    #aic_vals.append(km.aic(X_test_next))
    #bic_vals.append(km.bic(X_test_next))
    rmse_vals.append(mean_squared_error(y_test_next, pred_y_test_next))


    #acc_scores.append(accuracy_score(y_test, y_pred))

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('Number of clusters')
plt.title('Performance evaluation scores')
plt.grid(True)
plt.savefig('perf km and ica.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('var km and ica.png')



fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('rmse km and ica.png')

fig2,ax2 = plt.subplots()
ax2.plot(num_components, acc_scores)
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('accuracy score')
plt.savefig('km and ica acc.png')  #std of recon error for rp
plt.clf()

homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []
acc_scores = []
for c in range(3, X_train.shape[1]+1):
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    ica = FastICA(n_components = c, random_state=2)
    #pca = pca.fit(X_train)

    principalComponents = ica.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    #print(principalDf)


    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    finalDf = clean_dataset(finalDf)    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)

    em = GaussianMixture(n_components=c, random_state=2).fit(X_train_next)#KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = em.predict(X_test_next)

    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)

    comp_score = metrics.completeness_score(y_test_next, pred_y_test_next)
    homo_score = metrics.homogeneity_score(y_test_next, pred_y_test_next)
    var_score = em.score(X_test_next)
    sil_score = metrics.silhouette_score(X_test_next, pred_y_test_next, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    aic_vals.append(em.aic(X_test_next))
    bic_vals.append(em.bic(X_test_next))
    rmse_vals.append(mean_squared_error(y_test_next, pred_y_test_next))


    #acc_scores.append(accuracy_score(y_test, y_pred))

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('Number of clusters')
plt.title('Performance evaluation scores')
plt.grid(True)
plt.savefig('perf em and ica.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('var em and ica.png')


fig5, ax5 = plt.subplots()
ax5.plot(num_components, aic_vals)
ax5.plot(num_components, bic_vals)
plt.title('AIC/BIC values')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('aic em and ica.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('rmse em and ica.png')

fig2,ax2 = plt.subplots()
ax2.plot(num_components, acc_scores)
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('accuracy score')
plt.savefig('em and ica acc.png')  #std of recon error for rp






homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []
acc_scores = []
for c in range(3, X_train.shape[1]+1):
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    #ica = FastICA(n_components = c, random_state=2)
    #pca = pca.fit(X_train)
    rand = random_projection.GaussianRandomProjection(n_components = c)

    principalComponents = rand.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)


    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)

    km = KMeans(n_clusters=c, random_state=2).fit(X_train_next)#KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = km.predict(X_test_next)
    print(pred_y_test_next)

    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)

    comp_score = metrics.completeness_score(y_test_next, pred_y_test_next)
    homo_score = metrics.homogeneity_score(y_test_next, pred_y_test_next)
    var_score = km.score(X_test_next)
    sil_score = metrics.silhouette_score(X_test_next, pred_y_test_next, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    #aic_vals.append(km.aic(X_test_next))
    #bic_vals.append(km.bic(X_test_next))
    rmse_vals.append(mean_squared_error(y_test_next, pred_y_test_next))


    #acc_scores.append(accuracy_score(y_test, y_pred))

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('Number of clusters')
plt.title('Performance evaluation scores')
plt.grid(True)
plt.savefig('perf km and rand.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('var km and rand.png')



fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('rmse km and rand.png')

fig2,ax2 = plt.subplots()
ax2.plot(num_components, acc_scores)
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('accuracy score')
plt.savefig('km and rand acc.png')  #std of recon error for rp
plt.clf()

homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []
acc_scores = []
for c in range(3, X_train.shape[1]+1):
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    rand = random_projection.GaussianRandomProjection(n_components = c)
    #pca = pca.fit(X_train)

    principalComponents = rand.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    #print(principalDf)


    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)
    finalDf = clean_dataset(finalDf)
    print(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)

    em = GaussianMixture(n_components=c, random_state=2).fit(X_train_next)#KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = em.predict(X_test_next)

    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)

    comp_score = metrics.completeness_score(y_test_next, pred_y_test_next)
    homo_score = metrics.homogeneity_score(y_test_next, pred_y_test_next)
    var_score = em.score(X_test_next)
    sil_score = metrics.silhouette_score(X_test_next, pred_y_test_next, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    aic_vals.append(em.aic(X_test_next))
    bic_vals.append(em.bic(X_test_next))
    rmse_vals.append(mean_squared_error(y_test_next, pred_y_test_next))


    #acc_scores.append(accuracy_score(y_test, y_pred))

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('Number of clusters')
plt.title('Performance evaluation scores')
plt.grid(True)
plt.savefig('perf em and rand.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('var em and rand.png')


fig5, ax5 = plt.subplots()
ax5.plot(num_components, aic_vals)
ax5.plot(num_components, bic_vals)
plt.title('AIC/BIC values')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('aic em and rand.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('rmse em and rand.png')

fig2,ax2 = plt.subplots()
ax2.plot(num_components, acc_scores)
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('accuracy score')
plt.savefig('em and rand acc.png')  #std of recon error for rp




homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []
acc_scores = []
for c in range(3, X_train.shape[1]+1):
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    #ica = FastICA(n_components = c, random_state=2)
    #pca = pca.fit(X_train)
    lin = LinearDiscriminantAnalysis(n_components = c)

    principalComponents = lin.fit_transform(X, y)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    ##print(principalDf)


    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    finalDf = clean_dataset(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)

    km = KMeans(n_clusters=c, random_state=2).fit(X_train_next)#KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = km.predict(X_test_next)
    print(pred_y_test_next)

    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)

    comp_score = metrics.completeness_score(y_test_next, pred_y_test_next)
    homo_score = metrics.homogeneity_score(y_test_next, pred_y_test_next)
    var_score = km.score(X_test_next)
    sil_score = metrics.silhouette_score(X_test_next, pred_y_test_next, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    #aic_vals.append(km.aic(X_test_next))
    #bic_vals.append(km.bic(X_test_next))
    rmse_vals.append(mean_squared_error(y_test_next, pred_y_test_next))


    #acc_scores.append(accuracy_score(y_test, y_pred))

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('Number of clusters')
plt.title('Performance evaluation scores')
plt.grid(True)
plt.savefig('perf km and lin.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('var km and lin.png')



fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('rmse km and lin.png')

fig2,ax2 = plt.subplots()
ax2.plot(num_components, acc_scores)
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('accuracy score')
plt.savefig('km and lin acc.png')  #std of recon error for rp
plt.clf()

homo_vals =[]
comp_vals = []
sil_vals = []
var_vals = []
num_components = []
aic_vals = []
bic_vals = []
rmse_vals = []
acc_scores = []
for c in range(3, X_train.shape[1]+1):
    #km = KMeans(n_clusters=c, random_state=0).fit(X_train)
    rand = LinearDiscriminantAnalysis(n_components = c)
    #pca = pca.fit(X_train)

    principalComponents = lin.fit_transform(X, y)
    principalDf = pd.DataFrame(data = principalComponents)
    #y_pred = pca.predict(X_test)
    #print(principalDf)


    finalDf = pd.concat([principalDf, dataset[['quality']]], axis = 1)

    finalDf = clean_dataset(finalDf)

    print(finalDf)
    

    x_next = finalDf.iloc[:,:-1]
    y_next = finalDf.iloc[:, -1].values

    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(x_next, y_next, test_size = 0.3)

    em = GaussianMixture(n_components=c, random_state=2).fit(X_train_next)#KMeans(n_clusters=c, random_state=0).fit(X_train_next)

    
    #kmeans.fit(X_train)

    pred_y_test_next = em.predict(X_test_next)

    acc_scores.append(accuracy_score(y_test_next, pred_y_test_next))

    #print( acc_scores)

    comp_score = metrics.completeness_score(y_test_next, pred_y_test_next)
    homo_score = metrics.homogeneity_score(y_test_next, pred_y_test_next)
    var_score = em.score(X_test_next)
    sil_score = metrics.silhouette_score(X_test_next, pred_y_test_next, metric='euclidean')
    homo_vals.append(homo_score)
    comp_vals.append(comp_score)
    sil_vals.append(sil_score)
    var_vals.append(var_score)
    num_components.append(c)
    aic_vals.append(em.aic(X_test_next))
    bic_vals.append(em.bic(X_test_next))
    rmse_vals.append(mean_squared_error(y_test_next, pred_y_test_next))


    #acc_scores.append(accuracy_score(y_test, y_pred))

fig4,ax4 = plt.subplots()
ax4.plot(num_components, homo_vals)
ax4.plot(num_components, comp_vals)
ax4.plot(num_components, sil_vals)
plt.legend(['homogenity','completeness','silhoutette'])
plt.xlabel('Number of clusters')
plt.title('Performance evaluation scores')
plt.grid(True)
plt.savefig('perf em and lin.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, var_vals)
plt.title('Variance explained by each cluster')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('var em and lin.png')


fig5, ax5 = plt.subplots()
ax5.plot(num_components, aic_vals)
ax5.plot(num_components, bic_vals)
plt.title('AIC/BIC values')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('aic em and lin.png')

fig5, ax5 = plt.subplots()
ax5.plot(num_components, rmse_vals)
plt.title('Mean Squared Error')
plt.xlabel('Number of clusters')
plt.grid(True)
plt.savefig('rmse em and lin.png')

fig2,ax2 = plt.subplots()
ax2.plot(num_components, acc_scores)
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('accuracy score')
plt.savefig('em and lin acc.png')  #std of recon error for rp
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

dim_frame = PCA(random_state = 2)
X_PCA = dim_frame.fit_transform(X)

k_range = np.arange(1, X.shape[1]+1)
ax2.bar(k_range, dim_frame.explained_variance_ratio_, color='cyan')


plt.tight_layout()
plt.savefig('test3.png')




pca = PCA(n_components = 2)
principal_components = pca.fit_transform(X_test)
principalDf = pd.DataFrame(data = principal_components)
temp = pd.Dataframe(dataset.iloc[:, -1].values)
finalDf = pd.concat([principalDf, temp], axis = 1)

print(finalDf)



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
"""