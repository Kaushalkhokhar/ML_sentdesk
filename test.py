import pandas as pd 
from collections import Counter

df = {'Low':[1,2,3,4],
      'High':[5,6,7,8]}

df = pd.DataFrame(df)
#df = df[['Low']]
df.loc['med'] = [4,5]
#print(df)

x = [i*2 for i in range(len(df))]
# print(x)

x =[1, 1, 3, 5, 7, 7]

# print(sorted(x))

print(Counter(x))
print(Counter(x).most_common())