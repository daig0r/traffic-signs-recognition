import pandas as pd

from keras import preprocessing

DIR = 'data/images'

data_list = []
for  num in range(7):
    DIR_PATH = '{}/{}'.format(DIR, num)
    CSV_PATH = '{}/GT-{}.csv'.format(DIR_PATH, num)
    df = pd.read_csv(CSV_PATH, sep=';')
    df['Filename'] = DIR_PATH + '/' + df['Filename']
    df['ClassId'] =  num
    data_list.append(df)

data = pd.concat(data_list, ignore_index=True)

data.to_csv('data.csv', index=False)
