import pandas as pd
import numpy as np

data = pd.read_csv('./data.csv',encoding='cp949', index_col=0)
data =data.loc[:,data.loc['Item Name',:]!='공매도거래량(주)']
data_set = data[pd.notnull(data['A005930'])]

# print(data_set)
# print(data_set.Symbol)
newColumns = data_set.loc['Symbol Name',:] + '_' + data_set.loc['Item Name',:]
data_set.columns = newColumns
data_set = data_set.drop(data_set.index[0:2])
dfs = []
for i in range(201):
    if not pd.isnull(data_set.ix[0,7*i]):
        df = data_set.ix[:,7*i:7*(i+1)]
        df_names = df.columns[0].split('_')[0]
        dfs.append(df_names)
        df.to_csv('./data/%s.csv'%df_names)

dd = pd.DataFrame(dfs, columns=['종목'])
dd.to_csv('./data/종목.csv')