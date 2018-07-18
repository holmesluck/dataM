import numpy as np
import pandas as pd
import pandas_datareader.data as web
import math
import matplotlib.pyplot as plt

# yahoo dax data
dax = web.DataReader(name='^GDAXI',data_source='yahoo',start = '2000-1-1')

# print out
print(dax.info())

dax['Close'].plot(figsize=(8,5))
plt.show()
