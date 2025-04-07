import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

src="192.168.74.10"
filename = "./datasciense/data/conn.dat"
df = pd.read_csv(filename, sep='\t')

today = datetime(2014, 3, 25).date()    # example for experiment

# time range: 2011-07-22, 2014-03-24
startday = today - timedelta(days=977)
endday = today - timedelta(days=1)
print(startday, endday)

# print(df.head())

# print(df.shape) # (197505, 21)

# print(df.columns.tolist())
"""
['Unnamed: 0', 'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration', 
'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes',
'resp_pkts', 'resp_ip_bytes', 'tunnel_parents']
null data: local_orig
"""
# df = df.drop(columns=['Unnamed: 0', 'datasource'])
df = df[['id.orig_h', 'id.resp_h', 'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'conn_state', 'ts']]
print(df.head())

print(df.isnull().sum())

# Replace missing values
df['duration'].fillna('0', inplace = True)
df['orig_bytes'].fillna('0', inplace = True)
df['orig_bytes'].fillna('0', inplace = True)

print(df.isnull().sum())

print(df.dtypes)
# convert 'duration...' type to 'int'
df['duration'], df['orig_bytes'],df['resp_bytes']  = df['duration'].apply(lambda x: int(x)), df['orig_bytes'].apply(lambda x: int(x)), df['orig_bytes'].apply(lambda x: int(x))
df['orig_pkts'], df['resp_pkts'] = df['orig_pkts'].apply(lambda x: int(x)), df['resp_pkts'].apply(lambda x: int(x))

# convert 'conn_state' type to 'str'
df['conn_state'] = df['conn_state'].astype(str)

print(df.dtypes)

df['ts'] = pd.to_datetime(df['ts'], unit='s') # time range: 2011-07-22, 2014-03-24
df['date'] = df['ts'].apply(lambda x: x.date())

print(df.head())

datetime_sorted_unique_index = df['date'].drop_duplicates().sort_values().tolist()
print(datetime_sorted_unique_index, len(datetime_sorted_unique_index)) # 736

print(df.head())

df['id.orig_h'].value_counts() # src : 742; dest: 30939; src = 192.168.74.10

df = df[df['id.orig_h']==src]

df1 = df[['duration', 'orig_bytes', 'resp_bytes',  'orig_pkts',  'resp_pkts', 'date']]
df2 = df[['id.resp_h', 'date']]
df3 = df[['conn_state', 'date']]

# group by date
df1_1 = df1.groupby(['date']).sum()
df1_2 = df1.groupby(['date']).mean()
df1_3 = df1.groupby(['date']).max()

df1_f = pd.DataFrame(index=df1_1.index)

df1_f['duration_tot'], df1_f['duration_ave'], df1_f['duration_max'] = df1_1['duration'], df1_2['duration'], df1_3['duration']
df1_f['orig_bytes_tot'], df1_f['orig_bytes_ave'], df1_f['orig_bytes_max'] = df1_1['orig_bytes'], df1_2['orig_bytes'], df1_3['orig_bytes']
df1_f['resp_bytes_tot'], df1_f['resp_bytes_ave'], df1_f['resp_bytes_max'] = df1_1['resp_bytes'], df1_2['resp_bytes'], df1_3['resp_bytes']
df1_f['orig_pkts_tot'], df1_f['orig_pkts_ave'], df1_f['orig_pkts_max'] = df1_1['orig_pkts'], df1_2['orig_pkts'], df1_3['orig_pkts']
df1_f['resp_pkts_tot'], df1_f['resp_pkts_ave'], df1_f['resp_pkts_max'] = df1_1['resp_pkts'], df1_2['resp_pkts'], df1_3['resp_pkts']
print(df1_f.head())

df2_f = df2.groupby(['date']).nunique()
print(df2_f.head())


print(df1_3.head())
datetime_sorted_unique_index = df['date'].drop_duplicates().sort_values().tolist()
# len(datetime_sorted_unique_index), datetime_sorted_unique_index

# all possible sig_ids
conn_states = ['S0','S1','REJ']

dates = []
startDate = startday
endDate = endday
addDays = timedelta(days=1)
while startDate <= endDate:
    dates.append(startDate)
    startDate += addDays

new_df3 = pd.DataFrame(columns = conn_states, index = datetime_sorted_unique_index)

startDate = startday
endDate = endday
addDays = timedelta(days=1)

while startDate <= endDate:
    day = startDate   
    if day in datetime_sorted_unique_index:
        filt_df = df[df['date'] == day]
        cnts = filt_df['conn_state'].value_counts()
        for x in conn_states:
            if x in cnts.keys():
                new_df3.loc[day, x] = cnts[x]
            else:
                new_df3.loc[day, x] = 0
        
    startDate += addDays

pd.set_option('max_columns', None)
new_df3.head()

print(new_df3.shape)

print(new_df3.head())

df3_f = new_df3.copy()

# combine df1, df2
new_df = pd.concat([df2_f, df1_f, df3_f], axis=1)

print(new_df.head())

print(new_df.shape) # (31, 19)

file_name = "./datasciense/results/{}_processed_data_src.csv".format(src)
# new_df.to_csv(file_name, index=True)
new_df.to_csv(file_name, index=True, date_format='%Y-%m-%d')


# Plot distribution before transform
X_plot = new_df.copy()
plt.rcParams.update({'font.size': 6})
figure, axis = plt.subplots(5, 4)
for i in range(len(X_plot.columns)):
    X_plot[X_plot.columns[i]].plot(ax=axis[i//4,i%4], kind='hist', bins=100, title=X_plot.columns[i])
plt.tight_layout()
file_name = "./datasciense/results/{}_distribution_before_transform_plot.svg".format(src)
plt.savefig(file_name)
plt.show()



