import pandas as pd
import torch 
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

filepath = './data/Gift_Cards.jsonl'
df = pd.read_json(filepath, lines=True)
print(df.head(5))

def preprocess(data):
    df['user_id_encoded'] = LabelEncoder(df['user_id'])
    df['asin_encoded'] = LabelEncoder(df['asin'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['day_of_week'] = df['timestamp'].dt.day_of_week
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    
    df['verified_purchase'] = df['verified_purchase'].astype(int)
    
    scaler = MinMaxScaler()
    df['rating'] = scaler.fit_transform(df['rating'])
    df['helpful_vote'] = scaler.fit_transform(df['helpful_vote'])
    