import os
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from feature_generation import feature_generation
from data_preprocessing import main_preprocessing
from bert_inference import bert_inference_pipeline

## Constant variables
# Data 
ATTRIBUTES_TEST_PATH = Path('./data/test/attributes_test.parquet')
RESNET_PATH = Path('./data/test/resnet_test.parquet')
TEXT_AND_BERT_PATH = Path('./data/test/text_and_bert_test.parquet')
VAL_PATH = Path('./data/test/test.parquet')

#Catboost
CATBOOST_MODEL_PATH = Path('./models/CATBOOST/')

cat_features = [
    'category_level_1_match', 'category_level_2_match',
    'category_level_3_match','category_level_4_match',
    'description_match','brand_match', 'country_match',
    'main_pic_embeddings_resnet_v1_fillness', 'category_level_2'
]

def load_test_data():

    attributes = pd.read_parquet(ATTRIBUTES_TEST_PATH, engine='pyarrow')
    resnet = pd.read_parquet(RESNET_PATH, engine='pyarrow')
    text_and_bert = pd.read_parquet(TEXT_AND_BERT_PATH, engine='pyarrow')
    test = pd.read_parquet(VAL_PATH, engine='pyarrow')

    return attributes, resnet, text_and_bert, test

def generate_features(
    attributes: pd.DataFrame,
    resnet: pd.DataFrame,
    text_and_bert: pd.DataFrame,
    test: pd.DataFrame) -> pd.DataFrame:

    data = pd.concat([attributes, resnet.drop('variantid', axis=1), text_and_bert.drop('variantid', axis=1)], axis=1)
    data['description'] = data['description'].fillna('no desc')
    data = main_preprocessing(data)
    test_features_df = feature_generation(data, test)

    return test_features_df

def make_prediction(test_feature_df: pd.DataFrame) -> pd.DataFrame:

    def get_predictions_catboost(test_feature_df, submission_df):

        categories = test_feature_df['category_level_2'].unique()
        test_feature_df[cat_features] = test_feature_df[cat_features].astype('str')
        pred_proba_common_model = np.zeros(len(test_feature_df))

        for model in os.listdir(CATBOOST_MODEL_PATH):
            if model == 'categories':
                continue
            model_for_all_cb = CatBoostClassifier().load_model(CATBOOST_MODEL_PATH / model)
            predicted_proba = model_for_all_cb.predict_proba(test_feature_df)[:, 1]
            pred_proba_common_model += predicted_proba
        
        pred_proba_common_model /= 5
        submission_df.loc[:, 'predicted_proba_model_all_cb'] = pred_proba_common_model

        for category in categories:  
            x_indice_mask = test_feature_df.category_level_2 == category
            cat_pred_proba_array = np.zeros(len(test_feature_df[test_feature_df.category_level_2 == category]))

            for model in os.listdir(CATBOOST_MODEL_PATH / 'categories' / category):
        
                model = CatBoostClassifier().load_model(CATBOOST_MODEL_PATH / 'categories' / category / model)
                predicted_proba = model.predict_proba(test_feature_df.loc[x_indice_mask, :])[:, 1]
                cat_pred_proba_array += predicted_proba
            
            cat_pred_proba_array /= 5
            submission_df.loc[x_indice_mask, 'predicted_proba_category_catboost'] = cat_pred_proba_array

        return submission_df

    columns_to_drop = ['variantid_1', 'variantid_2', 'category_level_1']

    submission_df = test_feature_df[['variantid_1', 'variantid_2']]
    submission_df.loc[:, 'predicted_proba_category_catboost'] = [0 for i in range(len(test_feature_df))]

    test_feature_df.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    submission_df = get_predictions_catboost(test_feature_df.copy(), submission_df)

    target = 0.6 * submission_df['predicted_proba_model_all_cb'] + 0.4 * submission_df['predicted_proba_category_catboost'] 
    submission_df.loc[:, 'target'] = target
    
    submission_df.drop([
        'predicted_proba_model_all_cb', 
        'predicted_proba_category_catboost',
        ], axis=1, inplace=True)

    return submission_df

def return_csv(submission_df):
    submission_df = submission_df.rename(
        columns={
            'variantid_1': 'variantid1',
            'variantid_2': 'variantid2'
        }
    )

    submission_df.to_csv('./data/submission.csv', index=False)

def main():
    attributes, resnet, text_and_bert, test = load_test_data()
    name_attr_bert_df = bert_inference_pipeline(text_and_bert.copy(), attributes.copy(), test.copy())
    test_feature_df = generate_features(attributes.copy(), resnet.copy(), text_and_bert.copy(), test.copy())
    test_feature_df = pd.concat(
        [
            test_feature_df.copy(), 
            name_attr_bert_df.copy().drop(columns=['variantid_1', 'variantid_2'], axis=1)
        ], 
        axis=1
    )
    submission_df = make_prediction(test_feature_df)

    return_csv(submission_df)

if __name__ == '__main__':
    main()

