import pandas as pd 
df = {'Low':[1,2,3,4],
      'High':[5,6,7,8]}

df = pd.DataFrame(df)
#df = df[['Low']]
df.loc['med'] = [4,5]
print(df)

x = [i*2 for i in range(len(df))]
print(x)