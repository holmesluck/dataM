# import csv
#
# data = []
# with open("./resource/test.csv",newline='') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         data.append(row)
#
# print(data)
import pandas as pd
import scipy
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import jieba
from PIL import Image
import wordcloud as wc

import seaborn as sns; sns.set()
# read data and do some data transfer

testdata = open("./resource/test.csv","r",encoding="UTF-8").read()

data = pd.read_csv('./resource/test.csv')
# print(data)
AS_mapping = {"Large":2,"Medium":1,"Small":0}
data['AreaSize'] = data['AreaSize'].map(AS_mapping)
testa = np.array(data[-1:])


#check the missing value
missper = data.describe()
# print(missper)


# normalization function for the data
def MaxMinNormalization(x):
    x = (x - np.min(x))/ (np.max(x) - np.min(x))
    return x
# normalization
data['AgeOfStore'] = MaxMinNormalization(data['AgeOfStore'])
data['SalesInThousands'] = MaxMinNormalization(data['SalesInThousands'])
feature = ['AreaSize','AgeOfStore','Promotion','Month','SalesInThousands']
datatrain = data[feature]
# print(data[feature])
# specify the initial centroids


# count the times of the each element exist in one column
a = datatrain['AreaSize'].value_counts()
print(a)

# list all the skew for data
b = datatrain['AgeOfStore'].skew()
print(b)

# calculate the correlation of the two column
c = datatrain['AgeOfStore'].corr(datatrain['SalesInThousands'])
print(c)

# show in heat map for the data set
d = datatrain.corr()
f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(d,vmax=.9,square =True)


# show it using the clutermap
e = sns.clustermap(datatrain)




# try to use the kmeans to do the clustering
# model = KMeans(n_clusters=2 , init=data[], n_init=1)
# model.fit()
# no centroids kmeans
model = KMeans(n_clusters=1)
model.fit(data[feature])

# get the label
labels = model.predict(data[feature])

centroids = model.cluster_centers_

# print('centroids:',centroids)
# print('label:',labels)

# testing the convert the dataframe to string by using pandas and delet the index name
# # option 1 no use
# datatrain.rename_axis(None,1,inplace=True)
# print(datatrain)
# print("###############################")
# # option 2
# datatrain.columns = datatrain.columns.tolist()
# print(datatrain)
# print("###############################")
# # option 3
# datatrain.columns.name = None
# print(datatrain)
# print("###############################")
# # option 4 not success
# datatrain._reindex_columns(new_columns = None)
# print(datatrain)
# print("###############################")

# print(datatrain)
test = datatrain.to_string()
# print(test)
# print(type(test))

# creat the pic show out
fig = plt.gcf()
fig1 = plt.gcf()
fig2 = plt.gcf()
fig3 = plt.gcf()



# draw a scatter diagram
datapic = pd.concat([datatrain['SalesInThousands'],datatrain['AgeOfStore']],axis=1)
datapic.plot.scatter(x="AgeOfStore",y='SalesInThousands',xlim=(0,1),ylim= (0,1))
datapic1 = pd.concat([datatrain['SalesInThousands'],datatrain['Month']],axis=1)
datapic1.plot.scatter(x="Month",y='SalesInThousands',xlim=(0,5),ylim= (0,1))
datapic2 = pd.concat([datatrain['SalesInThousands'],datatrain['AreaSize']],axis=1)
datapic2.plot.scatter(x="AreaSize",y='SalesInThousands',ylim= (0,1))
datapic3 = pd.concat([datatrain['SalesInThousands'],datatrain['Promotion']],axis=1)
datapic3.plot.scatter(x="Promotion",y='SalesInThousands',ylim= (0,1))
datapic4 = pd.concat([datatrain['SalesInThousands'],datatrain['Promotion']],axis=1)
datapic4.plot.scatter(x="Promotion",y='SalesInThousands',ylim= (0,1))
# fig.savefig('./resource/scatterAgS.png',dpi=100)
# fig2.savefig('./resource/scatterAsS.png',dpi=100)
# fig1.savefig('./resource/scatterAmS.png',dpi=100)
# fig3.savefig('./resource/scatterApS.png',dpi=100)

plt.show()





