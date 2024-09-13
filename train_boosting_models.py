import mlflow
import pandas as pd
import numpy as np 

from catboost import Pool, CatBoostClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pathlib import Path
from tqdm.notebook import tqdm

DATA_FOLDER_PATH = Path('./data/train')

MAIN_MODEL_PATH = Path('./models/CATBOOST')
CATEGORY_MODEL_PATH = Path('./models/CATBOOST/categories')

cat_feature_list = [
    'category_level_1_match',
    'category_level_2_match',
    'category_level_3_match',
    'category_level_4_match',
    'description_match',
    'brand_match',
    'country_match',
    'main_pic_embeddings_resnet_v1_fillness',
    'category_level_2']

def get_hack_metric(X_eval, model):
    categories = X_eval['category_level_2'].unique()

    pr_auc_by_category = []
    pr_auc_dict = {}

    for category in categories:

        y_true = X_eval[X_eval['category_level_2'] == category].target
        y_scores = model.predict_proba(X_eval[X_eval['category_level_2'] == category].drop('target', axis=1))[:, 1]

        if len(y_true) == 0 or sum(y_true) == 0:
            pr_auc_by_category.append(0)
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        pr_auc = auc(recall, precision)
        pr_auc_by_category.append(pr_auc)
        pr_auc_dict[category] = pr_auc

    if len(pr_auc_by_category) == 0:
       return 0.0

    macro_prauc = np.mean(pr_auc_by_category)
    return macro_prauc, pr_auc_dict

def data_preparation():

    train_df = pd.read_parquet(DATA_FOLDER_PATH / 'train_feature_df.parquet', engine='pyarrow')
    attr_oof_features = pd.read_parquet(DATA_FOLDER_PATH / 'name_attr_bert_oof.parquet', engine='pyarrow')  #TODO FIX
    desc_oof_features = pd.read_parquet(DATA_FOLDER_PATH / 'name_desc_bert_oof.parquet', engine='pyarrow') #TODO FIX
    multi_attrs_oof = pd.read_parquet(DATA_FOLDER_PATH / 'multi_attr_bert_oof.parquet', engine='pyarrow') #TODO FIX

    train_df['brand_match'] = train_df['brand_match'].astype('str')
    train_df['country_match'] = train_df['brand_match'].astype('str')

    attr_oof_features.rename(
    columns={
        'variantid1': 'variantid_1',
        'variantid2': 'variantid_2'
        }, inplace=True
    )

    desc_oof_features.rename(
        columns={
            'variantid1': 'variantid_1',
            'variantid2': 'variantid_2',
            'name_attr_bert_oof': 'name_desc_bert_oof'
        }, inplace=True
    )

    multi_attrs_oof.rename(
        columns={
            'variantid1': 'variantid_1',
            'variantid2': 'variantid_2'
        }, inplace=True
    )

    train_df = train_df[train_df['category_level_2_1'] == train_df['category_level_2_2']]
    train_df['category_level_2'] = train_df['category_level_2_1']
    train_df = train_df[
        ~train_df['category_level_2'].isin(
            ['Антиквариат и коллекционирование', 'Ювелирные изделия', 'Автомототехника', 'Фермерское хозяйство']
        )
    ]

    train_df = pd.merge(train_df, attr_oof_features.drop(columns=['target'], axis=1), on=['variantid_1', 'variantid_2'])
    train_df = pd.merge(train_df, desc_oof_features.drop(columns=['target'], axis=1), on=['variantid_1', 'variantid_2'])
    train_df = pd.merge(train_df, multi_attrs_oof.drop(columns=['target'], axis=1), on=['variantid_1', 'variantid_2'])

    train_df = train_df.drop(
        columns=[ 
            'variantid_1', 'variantid_2', 
            'category_level_1_1', 'category_level_1_2',
            'category_level_2_1', 'category_level_2_2',
            'category_level_3_1', 'category_level_3_2',
            'category_level_4_1', 'category_level_4_2',
        ], axis=1
    )

    return train_df

def train_catboost_solo(model_params, fitting_params, model_postfix, y_eval, X_eval, X_test, y_test):

    model = CatBoostClassifier(**model_params)

    eval_pool = fitting_params['eval_set']
    test_pool = fitting_params['test_set']
    fitting_params.pop('test_set', None)

    model.fit(**fitting_params)
    
    y_eval_pred_proba = model.predict_proba(eval_pool)[:, 1]
    predictions_val = model.predict(eval_pool)
    precision, recall, _ = precision_recall_curve(y_eval, y_eval_pred_proba)
    pr_auc_eval = auc(recall, precision)
    roc_auc_score_eval = roc_auc_score(y_eval, y_eval_pred_proba)
    f1_score_eval = f1_score(y_eval, predictions_val)
    
    predictions_test = model.predict(test_pool)
    y_test_pred_proba = model.predict_proba(test_pool)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
    pr_auc_test = auc(recall, precision)
    roc_auc_score_test = roc_auc_score(y_test, y_test_pred_proba)
    f1_score_test = f1_score(y_test, predictions_test)

    hack_metrics_eval, dict_metric_eval = get_hack_metric(X_eval, model)
    hack_metrics_test, dict_metric_test = get_hack_metric(X_test, model)

    mlflow.log_params(model_params)
    
    mlflow.log_metric("pr_auc_eval", pr_auc_eval)
    mlflow.log_metric("f1_score", f1_score_eval)
    mlflow.log_metric("ROC AUC", roc_auc_score_eval)
    
    mlflow.log_metric("pr_auc_test", pr_auc_test)
    mlflow.log_metric("f1_score_test", f1_score_test)
    mlflow.log_metric("ROC AUC test", roc_auc_score_test)

    mlflow.log_metric("Hack_metric_eval", hack_metrics_eval)
    mlflow.log_metric("Hack_metric_test", hack_metrics_test)

    model.save_model(MAIN_MODEL_PATH / f"catboost_model_{model_postfix}.cbm")

    mlflow.catboost.log_model(
        artifact_path=f"catboost_model_{model_postfix}.cbm", 
        cb_model=model,
        input_example=X_eval.drop('target', axis=1)[:10])
    
    return model, hack_metrics_eval, hack_metrics_test, dict_metric_eval, dict_metric_test

def get_common_catboost_model(train_df:pd.DataFrame):
    
    X = train_df
    X['brand_match'] = X['brand_match'].astype('str')
    y = train_df.target

    for seed in tqdm([42, 52, 228, 666, 1488]):

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

        train_pool = Pool(X_train.drop('target', axis=1), y_train, cat_features=cat_feature_list)
        eval_pool = Pool(X_eval.drop('target', axis=1), y_eval, cat_features=cat_feature_list)
        test_pool = Pool(X_test.drop('target', axis=1), y_test, cat_features=cat_feature_list)

        model_params = {
            'iterations': 50000, 
            'task_type': "GPU",
            'random_seed': seed
            }

        fitting_params = {
            'X': train_pool,
            'eval_set': eval_pool,
            'test_set': test_pool,
            'early_stopping_rounds':200,
            'verbose':1000
        }

        mlflow.set_experiment("Ozon_hack")

        with mlflow.start_run(run_name="Final Submission", description=''):
            model, hack_metrics_eval, hack_metrics_test, dict_metric_eval, dict_metric_test = train_catboost_solo(
                model_params, 
                fitting_params, 
                f"final_model_seed_{seed}",
                y_eval, X_eval, X_test, y_test)

def train_catboost_category(model_params, fitting_params, model_postfix, y_eval, X_eval, X_test, y_test, category):

    model = CatBoostClassifier(**model_params)

    eval_pool = fitting_params['eval_set']
    test_pool = fitting_params['test_set']
    fitting_params.pop('test_set', None)

    model.fit(**fitting_params)
    
    y_eval_pred_proba = model.predict_proba(eval_pool)[:, 1]
    predictions_val = model.predict(eval_pool)
    precision, recall, _ = precision_recall_curve(y_eval, y_eval_pred_proba)
    pr_auc_eval = auc(recall, precision)
    roc_auc_score_eval = roc_auc_score(y_eval, y_eval_pred_proba)
    f1_score_eval = f1_score(y_eval, predictions_val)
    
    predictions_test = model.predict(test_pool)
    y_test_pred_proba = model.predict_proba(test_pool)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
    pr_auc_test = auc(recall, precision)
    roc_auc_score_test = roc_auc_score(y_test, y_test_pred_proba)
    f1_score_test = f1_score(y_test, predictions_test)

    mlflow.log_params(model_params)
    
    mlflow.log_metric("pr_auc_eval", pr_auc_eval)
    mlflow.log_metric("f1_score", f1_score_eval)
    mlflow.log_metric("ROC AUC", roc_auc_score_eval)
    
    mlflow.log_metric("pr_auc_test", pr_auc_test)
    mlflow.log_metric("f1_score_test", f1_score_test)
    mlflow.log_metric("ROC AUC test", roc_auc_score_test)

    model.save_model(CATEGORY_MODEL_PATH / 'category' / category / f"catboost_model_{model_postfix}.cbm")

    mlflow.catboost.log_model(
        artifact_path=f"catboost_model_{model_postfix}.cbm", 
        cb_model=model,
        input_example=X_eval[:10])
    
    return model, pr_auc_eval, pr_auc_test

def get_cat_catboost_models(train_df):
    
    categories = train_df.category_level_2.unique()
    metrics_eval, metrics_test = [], []
    
    dict_prauc_eval_cat = {}
    dict_prauc_test_cat= {}
    
    for category in categories:

        X = train_df[train_df.category_level_2 == category]
        y = X.target
        X = X.drop(['target'], axis=1)

        X['brand_match'] = X['brand_match'].astype('str')

        for seed in tqdm([42, 52, 228, 666, 1488]):
            
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

            train_pool = Pool(X_train, y_train, cat_features=cat_feature_list)
            eval_pool = Pool(X_eval, y_eval, cat_features=cat_feature_list)
            test_pool = Pool(X_test, y_test, cat_features=cat_feature_list)

            model_params = {
                'iterations': 10000, 
                'task_type': "GPU",
                'random_state': seed
            }
            
            fitting_params = {
                'X': train_pool,
                'eval_set': eval_pool,
                'test_set': test_pool,
                'early_stopping_rounds': 100,
                'verbose': False
            }

            with mlflow.start_run(
                run_name=f"{category}_{seed}", 
                description=f'Отдельный катбуст на категорию. {seed}'):
                model, metric_eval, metric_test = train_catboost_category(model_params, fitting_params, f"{category}", y_eval, X_eval, X_test, y_test, category)
                metrics_eval.append(metric_eval) 
                metrics_test.append(metric_test)
                dict_prauc_eval_cat[category] = metric_eval 
                dict_prauc_test_cat[category] = metric_test 

def main():

    train_df = data_preparation()
    get_common_catboost_model(train_df.copy())
    get_cat_catboost_models(train_df.copy())

if '__name__' == 'main':
    main()
