import pandas as pd 
from collections import Counter

df = {'Low':[1,2,3,4],
      'High':[5,6,7,8]}

df = pd.DataFrame(df)


for i in df:
      for data in df[i]:
            print(data)