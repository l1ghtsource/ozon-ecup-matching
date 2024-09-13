import pandas as pd
from pathlib import Path

from data_preprocessing import main_preprocessing
from feature_generation import feature_generation

DATA_FOLDER_PATH_TRAIN = Path('./data/train')

ATTRIBUTES_PATH = DATA_FOLDER_PATH_TRAIN / 'attributes.parquet'
RESNET_PATH = DATA_FOLDER_PATH_TRAIN / 'resnet.parquet'
TEXT_AND_BERT_PATH = DATA_FOLDER_PATH_TRAIN / 'text_and_bertt.parquet'
TRAIN_PATH = DATA_FOLDER_PATH_TRAIN / 'train.parquet'

def main():

    attributes = pd.read_parquet(ATTRIBUTES_PATH, engine='pyarrow')
    resnet = pd.read_parquet(RESNET_PATH, engine='pyarrow')
    text_and_bert = pd.read_parquet(TEXT_AND_BERT_PATH, engine='pyarrow')
    train = pd.read_parquet(TRAIN_PATH, engine='pyarrow')

    data = pd.concat([attributes, resnet.drop('variantid', axis=1), text_and_bert.drop('variantid', axis=1)], axis=1)
    data['description'] = data['description'].fillna('no desc')
    data = main_preprocessing(data, mode='train')
    train_features_df = feature_generation(data, train)

    train_features_df.to_parquet(DATA_FOLDER_PATH_TRAIN / 'train_feature_df.parquet', engine='pyarrow')

    # ДОБАВИТЬ СЮДА ВЫЗОВ ОБУЧЕНИЯ БЕРТОВ

if __name__ == '__main__':
    main()