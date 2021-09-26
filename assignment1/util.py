import pandas as pd
import numpy as np

def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    indices_or_sections = [int(train_split * len(df)), int((1 - test_split) * len(df))]
    
    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds, test_ds

def get_smaller_dataframe_pd(df, frac, random_state=12):
    df = df.dropna()
    df = df.sample(frac=frac, random_state=random_state)
    return df

def normalized(df, method='MEAN'):
    if method == 'MEAN':
        return (df-df.mean())/df.std()
    elif method == 'MIN_MAX':
        return (df-df.min())/(df.max()-df.min())
    else:
        return df

def accuracy(y, y_hat):
    return sum(y == y_hat)/ len(y)

def mse(y, y_hat):
    return sum(pow(abs(y - y_hat), 2)) / len(y)
