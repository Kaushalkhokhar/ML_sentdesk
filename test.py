import pandas as pd 
from collections import Counter
import numpy as np
df = {'Low':[1,2,3,4],
      'High':[5,6,7,8]}

df = pd.DataFrame(df)

data_dict = {}
# for i in df:
#       for data in df[i]:
#             print(data)

for i in range(5):
      #data_dict[np.linalg.norm] = [i, i]
      data_dict[np.array] = [i, i]


#norms = sorted([n for n in data_dict])
#print(len(norms))
print(data_dict)
       