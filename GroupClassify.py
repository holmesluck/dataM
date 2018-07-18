import pandas as pd

data1 = pd.read_csv('./resource/test.csv')
print(data1)


# use groupby to combine multiple features in one new feature
AMS = data1.groupby(['AreaID','Month','StoreID'],as_index=False)
print("#######AMS########")
print(AMS)
print(data1)

# combine AreaID and StoreID to AreaID and try to calculate the mean value
AA = data1[["AreaID","StoreID"]].groupby(['AreaID'],as_index = False).mean()
print("#######AA########")
print(AA)


# classify to some specific value
AS = data1[data1['AreaSize'] == "Small" ].groupby(['AreaID'],as_index = False)
print("#########AS#########")
print(AS)