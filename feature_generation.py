import numpy as np
import pandas as pd
import ast
import jellyfish
import textdistance
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

tqdm.pandas()


def cosine_sim(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def euc_dist(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return euclidean_distances(vec1, vec2)[0][0]


def pair_cos_sim(vecs1, vecs2):
    if vecs1 is None or vecs2 is None:
        return {'mean': None, 'median': None, 'min': None, 'max': None, 'std': None}

    sim = []
    for vec1 in vecs1:
        for vec2 in vecs2:
            sim.append(cosine_sim(vec1, vec2))

    sim_array = np.array(sim)
    return {
        'mean': np.mean(sim_array),
        'median': np.median(sim_array),
        'min': np.min(sim_array),
        'max': np.max(sim_array),
        'std': np.std(sim_array)
    }

def cross_cos_sim(vec1, vecs2):
    if vec1 is None or vecs2 is None:
        return {'mean': None, 'median': None, 'min': None, 'max': None, 'std': None}

    sim = []
    for vec2 in vecs2:
        sim.append(cosine_sim(vec1, vec2))

    sim_array = np.array(sim)
    return {
        'mean': np.mean(sim_array),
        'median': np.median(sim_array),
        'min': np.min(sim_array),
        'max': np.max(sim_array),
        'std': np.std(sim_array)
    }


def fillness(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    condition_both = df[f'{col_name}_1'].notna() & df[f'{col_name}_2'].notna()
    condition_none = df[f'{col_name}_1'].isna() & df[f'{col_name}_2'].isna()

    df[f'{col_name}_fillness'] = np.where(
        condition_both, 'both',
        np.where(condition_none, 'none', 'only one')
    )

    return df


def avg_fully_eq_attributes(d1, d2):
    if d1 is None or d2 is None:
        return None
    d1 = ast.literal_eval(d1)
    d2 = ast.literal_eval(d2)
    keys = set(d1) & set(d2)
    metrics = []
    for key in keys:
        metrics.append(set(d1[key]) == set(d2[key]))
    return np.mean(metrics)


def longest_common_prefix(str1, str2):
    if str1 is None or str2 is None:
        return None

    min_len = min(len(str1), len(str2))
    prefix_len = 0

    for i in range(min_len):
        if str1[i] == str2[i]:
            prefix_len += 1
        else:
            break

    return prefix_len / min_len if min_len != 0 else 0


def longest_common_subsequence(str1, str2):
    if str1 is None or str2 is None:
        return None

    len1, len2 = len(str1), len(str2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[len1][len2]
    return lcs_len / max(len1, len2) if max(len1, len2) != 0 else 0


def jaccard_similarity(list1, list2):
    if list1 is None or list2 is None:
        return None
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def overlap_coefficient(list1, list2):
    if list1 is None or list2 is None:
        return None
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    return intersection / min(len(set1), len(set2)) if min(len(set1), len(set2)) != 0 else 0


def feature_generation(data: pd.DataFrame, test_pairs: pd.DataFrame) -> pd.DataFrame:

    test_pairs.rename(
        columns={
            'variantid1': 'variantid_1',
            'variantid2': 'variantid_2'
        }, inplace=True
    )

    test_df = test_pairs.merge(data.add_suffix('_1'),  on='variantid_1').merge(data.add_suffix('_2'), on='variantid_2')
    test_df = match_category(test_df)
    test_df = description_match(test_df)
    test_df = match_brands(test_df)
    test_df = match_countries(test_df)
    test_df = length_relation(test_df)
    test_df = main_pic_embed_match(test_df)
    test_df = match_pic_embedd_resnet(test_df)
    test_df = match_main_pic_embeddings_resnet_pic_embeddings_resnet(test_df)
    test_df = match_bert(test_df)
    test_df = match_concat_emb(test_df)
    test_df = tfidf_match(test_df)
    test_df = analyze_fillness(test_df)
    test_df = match_attr_dict(test_df)
    test_df = title_match(test_df)
    test_df = find_lcp_lcs_name(test_df)
    test_df = list_similarity(test_df)

    test_df.drop(
        columns=[
            'name_1', 'name_2',
            'description_1', 'description_2',
            'name_norm_1', 'name_norm_2',
            'description_norm_1', 'description_norm_2',
            'attr_vals_1', 'attr_vals_2',
            'attr_keys_1', 'attr_keys_2',
            'characteristics_attributes_1', 'characteristics_attributes_2',
            'description_tokens_1', 'description_tokens_2',
            'name_tokens_1', 'name_tokens_2',
            'attr_vals_w_digits_1', 'attr_vals_w_digits_2',
            'description_tokens_w_digits_1', 'description_tokens_w_digits_2',
            'name_tokens_w_digits_1', 'name_tokens_w_digits_2',
            'name_tfidf_1', 'name_tfidf_2',
            'description_tfidf_1', 'description_tfidf_2',
            'attr_keys_tfidf_1', 'attr_keys_tfidf_2',
            'attr_vals_tfidf_1', 'attr_vals_tfidf_2',
            'name_bert_64_1', 'name_bert_64_2',
            'main_pic_embeddings_resnet_v1_1', 'main_pic_embeddings_resnet_v1_2',
            'pic_embeddings_resnet_v1_1', 'pic_embeddings_resnet_v1_2',
            'concat_emb_1', 'concat_emb_2',
            'characteristic_attributes_mapping_1', 'characteristic_attributes_mapping_2',
            'pic_embeddings_resnet_v1_len_1', 'pic_embeddings_resnet_v1_len_2',
            'name_tokens_len_1', 'name_tokens_len_2',
            'description_tokens_len_1', 'description_tokens_len_2',
            'characteristics_attributes_len_1', 'characteristics_attributes_len_2'
        ],
        axis=1,
        inplace=True
    )

    test_df['category_level_1'] = test_df['category_level_1_1']
    test_df['category_level_2'] = test_df['category_level_2_1']

    for i in range(1, 5):
        test_df.drop(
            columns=[f'category_level_{i}_1', f'category_level_{i}_2'],
            axis=1,
            inplace=True
        )

    return test_df

# Мэтч по категориям (полное совпадение + частичное по последнему уровню)
def match_category(df):
    for i in range(1, 5):
        df[f'category_level_{i}_match'] = df.progress_apply(
            lambda row: row[f'category_level_{i}_1'].lower() == row[f'category_level_{i}_2'].lower(), axis=1
        )
        if i == 4:
            df[f'category_level_{i}_token_sort_ratio_match'] = df.progress_apply(
                lambda row: fuzz.token_sort_ratio(row[f'category_level_{i}_1'],
                                                  row[f'category_level_{i}_2']) / 100, axis=1)
    return df

# Мэтч по описанию
def description_match(df):
    df[f'description_match'] = df.progress_apply(lambda row: row['description_1'] == row['description_2'], axis=1)
    return df

# Мэтч по бренду
def match_brands(df):
    def supportive_match(row):
        if row[f'brand_1'] is None or row[f'brand_2'] is None:
            return None
        return row[f'brand_1'].lower() == row[f'brand_2'].lower()

    df[f'brand_match'] = df.progress_apply(supportive_match, axis=1)
    df.drop(columns=['brand_1', 'brand_2'], axis=1, inplace=True)
    return df

# Мэтч по стране
def match_countries(df):

    def match_countries_for_row(row):
        if row[f'country_1'] is None or row[f'country_2'] is None:
            return None
        return row[f'country_1'].lower() == row[f'country_2'].lower()

    df[f'country_match'] = df.progress_apply(match_countries_for_row, axis=1)
    df.drop(columns=['country_1', 'country_2'], axis=1, inplace=True)
    return df

# Отношения длин
def length_relation(df):
    for col in ('pic_embeddings_resnet_v1_len', 'name_tokens_len', 'description_tokens_len',
                'characteristics_attributes_len'):
        df[f'{col}_ratio_left'] = df.progress_apply(
            lambda row: row[f'{col}_1'] / row[f'{col}_2'] if row[f'{col}_2'] not in (0, None) else 0, axis=1
        )
        df[f'{col}_ratio_right'] = df.progress_apply(
            lambda row: row[f'{col}_2'] / row[f'{col}_1'] if row[f'{col}_1'] not in (0, None) else 0, axis=1
        )

    for col in ('attr_vals', 'attr_vals_w_digits', 'attr_keys', 'name_tokens_w_digits'):
        df[f'{col}_ratio_left'] = df.progress_apply(
            lambda row: len(row[f'{col}_1'].split()) / len(row[f'{col}_2'].split())
            if len(row[f'{col}_2'].split()) not in (0, None) else 0, axis=1)
        df[f'{col}_ratio_right'] = df.progress_apply(
            lambda row: len(row[f'{col}_2'].split()) / len(row[f'{col}_1'].split())
            if len(row[f'{col}_1'].split()) not in (0, None) else 0, axis=1)
    return df

# Мэтч по main_pic_embeddings_resnet_v1 с main_pic_embeddings_resnet_v1
def main_pic_embed_match(df):
    df['main_pic_embeddings_resnet_v1_cos_sim'] = df.progress_apply(
        lambda row: cosine_sim(row['main_pic_embeddings_resnet_v1_1'], row['main_pic_embeddings_resnet_v1_2']), axis=1
    )
    df['main_pic_embeddings_resnet_v1_euc_dist'] = df.progress_apply(
        lambda row: euc_dist(row['main_pic_embeddings_resnet_v1_1'], row['main_pic_embeddings_resnet_v1_2']), axis=1
    )
    return df

# Мэтч по pic_embeddings_resnet_v1 с pic_embeddings_resnet_v1
def match_pic_embedd_resnet(df):
    results = df.progress_apply(
        lambda row: pair_cos_sim(row['pic_embeddings_resnet_v1_1'], row['pic_embeddings_resnet_v1_2']), axis=1
    )
    df['pic_embeddings_resnet_v1_mean_cos_sim'] = results.apply(lambda x: x['mean'])
    df['pic_embeddings_resnet_v1_median_cos_sim'] = results.apply(lambda x: x['median'])
    df['pic_embeddings_resnet_v1_min_cos_sim'] = results.apply(lambda x: x['min'])
    df['pic_embeddings_resnet_v1_max_cos_sim'] = results.apply(lambda x: x['max'])
    df['pic_embeddings_resnet_v1_std_cos_sim'] = results.apply(lambda x: x['std'])

    return df

# Мэтч по main_pic_embeddings_resnet_v1 с pic_embeddings_resnet_v1
def match_main_pic_embeddings_resnet_pic_embeddings_resnet(df):
    results = df.progress_apply(
        lambda row: cross_cos_sim(row['main_pic_embeddings_resnet_v1_1'], row['pic_embeddings_resnet_v1_2']), axis=1
    )
    df['cross1_mean_cos_sim'] = results.apply(lambda x: x['mean'])
    df['cross1_median_cos_sim'] = results.apply(lambda x: x['median'])
    df['cross1_min_cos_sim'] = results.apply(lambda x: x['min'])
    df['cross1_max_cos_sim'] = results.apply(lambda x: x['max'])
    df['cross1_std_cos_sim'] = results.apply(lambda x: x['std'])

    results = df.progress_apply(
        lambda row: cross_cos_sim(row['main_pic_embeddings_resnet_v1_2'], row['pic_embeddings_resnet_v1_1']), axis=1
    )
    df['cross2_mean_cos_sim'] = results.apply(lambda x: x['mean'])
    df['cross2_median_cos_sim'] = results.apply(lambda x: x['median'])
    df['cross2_min_cos_sim'] = results.apply(lambda x: x['min'])
    df['cross2_max_cos_sim'] = results.apply(lambda x: x['max'])
    df['cross2_std_cos_sim'] = results.apply(lambda x: x['std'])

    return df

# Мэтч по bert
def match_bert(df):
    df['name_bert_64_cos_sim'] = df.progress_apply(
        lambda row: cosine_sim(row['name_bert_64_1'], row['name_bert_64_2']), axis=1
    )
    df['name_bert_64_euc_dist'] = df.progress_apply(
        lambda row: euc_dist(row['name_bert_64_1'], row['name_bert_64_2']), axis=1
    )
    return df

# Мэтч по concat_emb
def match_concat_emb(df):
    df['concat_emb_cos_sim'] = df.progress_apply(
        lambda row: cosine_sim(row['concat_emb_1'], row['concat_emb_2']), axis=1
    )
    df['concat_emb_euc_dist'] = df.progress_apply(
        lambda row: euc_dist(row['concat_emb_1'], row['concat_emb_2']), axis=1
    )
    return df

# Мэтч по tfidf
def tfidf_match(df):
    for col in ('name', 'description', 'attr_keys', 'attr_vals'):
        df[f'{col}_tfidf_cos_sim'] = df.progress_apply(
            lambda row: cosine_sim(row[f'{col}_tfidf_1'], row[f'{col}_tfidf_2']), axis=1
        )
        df[f'{col}_tfidf_euc_dist'] = df.progress_apply(
            lambda row: euc_dist(row[f'{col}_tfidf_1'], row[f'{col}_tfidf_2']), axis=1
        )
    return df

# Заполненность строк у товаров
def analyze_fillness(df):
    df = fillness(df, 'main_pic_embeddings_resnet_v1')
    return df

# Совпадения для словаря атрибутов
def match_attr_dict(df):
    df['attributes_values_avg_fully_eq'] = (
        df.progress_apply(
            lambda row: avg_fully_eq_attributes(
                row['characteristic_attributes_mapping_1'],
                row['characteristic_attributes_mapping_2'],
            ),
            axis=1
        )
    )
    return df

# Мэтч по названию и атрибутам
def title_match(df):

    for col in ('name', 'name_norm', 'attr_vals', 'attr_keys', 'characteristics_attributes'):
        df[f'{col}_token_sort_ratio'] = df.progress_apply(
            lambda row: fuzz.token_sort_ratio(row[f'{col}_1'], row[f'{col}_2']) / 100, axis=1
        )
        df[f'{col}_token_set_ratio'] = df.progress_apply(
            lambda row: fuzz.token_set_ratio(row[f'{col}_1'], row[f'{col}_2']) / 100, axis=1
        )
        df[f'{col}_jaro_winkler_similarity'] = df.progress_apply(
            lambda row: jellyfish.jaro_winkler_similarity(row[f'{col}_1'], row[f'{col}_2']), axis=1
        )
        df[f'{col}_dice'] = df.progress_apply(
            lambda row: textdistance.dice(row[f'{col}_1'], row[f'{col}_2']), axis=1
        )
        df[f'{col}_tanimoto'] = df.progress_apply(
            lambda row: textdistance.tanimoto(row[f'{col}_1'], row[f'{col}_2']), axis=1
        )
        df[f'{col}_sorensen'] = df.progress_apply(
            lambda row: textdistance.sorensen(row[f'{col}_1'], row[f'{col}_2']), axis=1
        )
        if 'attr' not in col:
            df[f'{col}_damerau_levenshtein_distance'] = df.progress_apply(
                lambda row: jellyfish.damerau_levenshtein_distance(row[f'{col}_1'], row[f'{col}_2']), axis=1
            )
            df[f'{col}_WRatio'] = df.progress_apply(
                lambda row: fuzz.WRatio(row[f'{col}_1'], row[f'{col}_2']) / 100, axis=1
            )

    # Мэтч по описанию
    for col in ('description',):
        df[f'{col}_jaro_winkler_similarity'] = df.progress_apply(
            lambda row: jellyfish.jaro_winkler_similarity(row[f'{col}_1'], row[f'{col}_2']), axis=1
        )

    return df

# LCP + LCS для названий
def find_lcp_lcs_name(df):
    for col in ('name_norm',):
        df[f'{col}_lcp'] = df.progress_apply(
            lambda row: longest_common_prefix(row[f'{col}_1'], row[f'{col}_2']), axis=1
        )
        df[f'{col}_lcs'] = df.progress_apply(
            lambda row: longest_common_subsequence(row[f'{col}_1'], row[f'{col}_2']), axis=1
        )
    return df

# Сходство для списков
def list_similarity(df):
    for col in (
        'attr_keys',
        'attr_vals',
        'description_tokens',
        'name_tokens',
        'attr_vals_w_digits',
        'description_tokens_w_digits',
        'name_tokens_w_digits'
    ):
        df[f'{col}_jaccard_score'] = df.progress_apply(
            lambda row: jaccard_similarity(row[f'{col}_1'].split(), row[f'{col}_2'].split()), axis=1
        )
        df[f'{col}_overlap_score'] = df.progress_apply(
            lambda row: overlap_coefficient(row[f'{col}_1'].split(), row[f'{col}_2'].split()), axis=1
        )
    return df
