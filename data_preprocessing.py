import numpy as np
import pandas as pd
import json
import os 
import ast
import re
import pickle
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
tqdm.pandas()

# Нормализация текста
def normalize(text: str) -> str:
    text = text.lower()
    chars = []
    for char in text:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append(' ')
    tokens = ''.join(chars).split()
    return '_'.join(tokens)

# Удаление html тэгов и эмодзи из строки
def remove_html_tags_and_emoji(text):
    if text is None:
        return None
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.replace('\n', ' ')
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Нормализация названий товаров
def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    print('Нормализую названия товаров...')
    df['name'] = df['name'].progress_apply(remove_html_tags_and_emoji)
    df['name_norm'] = df['name'].progress_apply(normalize)
    df['name_tokens'] = df['name'].str.strip().str.lower()
    df['name'] = df['name_tokens'].progress_apply(lambda tokens: ' '.join(tokens.split()))
    return df

# Нормализация описаний товаров
def normalize_desc(df: pd.DataFrame) -> pd.DataFrame:
    print('Нормализую описания товаров...')
    df['description'] = df['description'].progress_apply(remove_html_tags_and_emoji)
    df['description_norm'] = df['description'].progress_apply(normalize)
    df['description_tokens'] = df['description'].str.strip().str.lower()
    df['description'] = df['description_tokens'].progress_apply(lambda tokens: ' '.join(tokens.split()))
    return df

# Выделение бренда как отдельной фичи
def extract_brand(df: pd.DataFrame) -> pd.DataFrame:
    print('Извлекаю названия брендов...')
    brand_arr = []
    for i in tqdm(range(len(df))):
        try:
            brand_arr.append(json.loads(df['characteristic_attributes_mapping'][i])['Бренд'][0])
        except:
            brand_arr.append(None)

    df['brand'] = brand_arr

    return df

# Выделение страны как отдельной фичи
def extract_country(df: pd.DataFrame) -> pd.DataFrame:
    print('Извлекаю страны-изготовители...')
    country_arr = []
    for i in tqdm(range(len(df))):
        try:
            country_arr.append(json.loads(df['characteristic_attributes_mapping'][i])['Страна-изготовитель'][0])
        except:
            country_arr.append(None)

    df['country'] = country_arr

    return df

# Извлечение категорий
def extract_categories(df: pd.DataFrame) -> pd.DataFrame:
    print('Извлекаю категории...')
    categories = pd.json_normalize(df['categories'].progress_apply(ast.literal_eval))
    categories.columns = [f'category_level_{i+1}' for i in range(categories.shape[1])]
    return df.drop(columns=['categories']).join(categories)

# [[emb]] -> [emb]
def squeeze_main_pic_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    print('Распаковываю эмбеддинги...')
    df['main_pic_embeddings_resnet_v1'] = df['main_pic_embeddings_resnet_v1'].progress_apply(
        lambda x: x[0] if isinstance(x, np.ndarray) else x
    )
    return df

# Нормализация атрибутов
def normalize_characteristic_attributes(df: pd.DataFrame) -> pd.DataFrame:
    def normalize_attributes(char_attrs_map):
        if char_attrs_map is not None:
            char_attrs_map = ast.literal_eval(char_attrs_map)
            parsed = [normalize(key) for key in char_attrs_map.keys()]
            return '; '.join([''.join(val) for val in parsed])
        return 'none'

    def normalize_values(attrs_map):
        if attrs_map is not None:
            attrs_map = ast.literal_eval(attrs_map)
            parsed = [list(map(normalize, attr_list)) for attr_list in attrs_map.values()]
            return '; '.join([' '.join(val) for val in parsed])
        return 'none'

    print('Нормализую значения атрибутов...')
    df['attr_vals'] = df['characteristic_attributes_mapping'].progress_apply(normalize_values)

    print('Нормализую атрибуты...')
    df['attr_keys'] = df['characteristic_attributes_mapping'].progress_apply(normalize_attributes)

    def combine_char_attributes(dct):
        if dct is not None:
            parsed = ast.literal_eval(dct)
            return '; '.join([f'{k}:{v}' for k, v in parsed.items()])
        return 'none'

    print('Собираю атрибуты и значения...')
    df['characteristics_attributes'] = df['characteristic_attributes_mapping'].progress_apply(combine_char_attributes)

    return df

# Количество картинок, токенов в названии, в описании
def get_lengths(df: pd.DataFrame) -> pd.DataFrame:
    def len_w_nans(x): return len(x) if x is not None else None

    print('Создаю количественные фичи...')
    df['pic_embeddings_resnet_v1_len'] = df['pic_embeddings_resnet_v1'].progress_apply(len_w_nans)
    df['name_tokens_len'] = df['name_tokens'].apply(lambda x: x.split()).progress_apply(len_w_nans)
    df['description_tokens_len'] = df['description_tokens'].apply(lambda x: x.split()).progress_apply(len_w_nans)
    df['characteristics_attributes_len'] = df['characteristics_attributes'].apply(
        lambda x: x.split('; ')).progress_apply(len_w_nans)

    return df

# Извлечение чисел из строк
def get_digits_elements(df: pd.DataFrame) -> pd.DataFrame:
    def has_more_than_two_digits(s):
        return len(re.findall(r'\d', s)) > 2

    print('Нахожу числа в названиях, описаниях и атрибутах...')
    for col in ('attr_vals', 'name_tokens', 'description_tokens'):
        if 'attr' not in col:
            df[f'{col}_w_digits'] = df[col].progress_apply(lambda row: ' '.join(
                [s for s in row.split() if has_more_than_two_digits(s)]))
        else:
            df[f'{col}_w_digits'] = df[col].progress_apply(lambda row: ' '.join(
                [s for s in row.split('; ') if has_more_than_two_digits(s)]))
    return df

# Конкатенированный эмбеддинг bert и resnet
def concat_embs(df: pd.DataFrame) -> pd.DataFrame:
    def normalize(array):
        norm = np.linalg.norm(array)
        if norm == 0:
            return array
        return array / norm

    print('Конкатенирую эмбеддинги...')
    df['concat_emb'] = df.progress_apply(
        lambda row: np.concatenate(
            [
                normalize(row['main_pic_embeddings_resnet_v1']),
                normalize(row['name_bert_64'])
            ]
        ),
        axis=1
    )
    return df


def load_tfidf_vectorizer(main_path, columns):
    tfidf_vectorizers = {}
    for col in columns:
        with open(f'{main_path}/{col}_tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        tfidf_vectorizers[col] = vectorizer
    return tfidf_vectorizers

def fit_tfidf_vectorizer(data, columns):
    tfidf_vectorizers = {}
    for col in columns:
        vectorizer = TfidfVectorizer()
        combined_texts = data[col].astype(str).tolist()
        vectorizer.fit(combined_texts)
        tfidf_vectorizers[col] = vectorizer
        try: 
            with open(f'./vectorizers/{col}_tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
        except FileNotFoundError: 
            os.mkdir('./vectorizers')
            with open(f'./vectorizers/{col}_tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)

    return tfidf_vectorizers

# Tfidf фичи
def tfidf_emb_gen(data, tfidf_vectorizers, columns, batch_size=5000):
    for col in columns:
        tfidf_col_sparse = []
        for start in tqdm(range(0, len(data), batch_size)):
            end = min(start + batch_size, len(data))
            batch_texts = data[col].iloc[start:end].astype(str).tolist()
            tfidf_batch_sparse = tfidf_vectorizers[col].transform(batch_texts)
            tfidf_col_sparse.append(tfidf_batch_sparse)
        tfidf_col_sparse = sp.vstack(tfidf_col_sparse)
        data[f'{col}_tfidf'] = [row for row in tfidf_col_sparse]
    return data


def main_preprocessing(data: pd.DataFrame, mode='test') -> pd.DataFrame:
    columns_for_tfidf = ['name', 'description', 'attr_keys', 'attr_vals']
    tf_ifd_vectorizer_path = './vectorizers'
    
    data = extract_categories(data)
    data = normalize_names(data)
    data = normalize_desc(data)
    data = extract_brand(data)
    data = extract_country(data)
    data = squeeze_main_pic_embeddings(data)
    data = normalize_characteristic_attributes(data)
    data = get_lengths(data)
    data = get_digits_elements(data)
    data = concat_embs(data)
    if mode == 'test':
        tf_idf_vectorizers = load_tfidf_vectorizer(main_path=tf_ifd_vectorizer_path, columns=columns_for_tfidf)
        data = tfidf_emb_gen(data, tfidf_vectorizers=tf_idf_vectorizers, columns=columns_for_tfidf)
    else: 
        data = tfidf_emb_gen(data, tfidf_vectorizers=fit_tfidf_vectorizer(data, columns_for_tfidf), columns=columns_for_tfidf)

    return data
