import re
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass

tqdm.pandas()

@dataclass
class Config:
    rubert_model: str = 'models/basemodel/rubert'
    distil_model: str = 'models/basemodel/distilbert'
    model_attr: str = 'models/BERT/3epoch_768_name_attr_bert_full.pth'
    model_attr_multi: str = 'models/BERT/multi512_attr_bert_full_second_epoch.pth'
    model_desc: str = 'models/BERT/3epoch_1024_name_desc_bert_full.pth'
    num_labels: int = 2
    max_len_attr: int = 768
    max_len_attr_multi: int = 512
    max_len_desc: int = 1024
    attr_cut: int = 1600
    attr_cut_multi: int = 1200
    desc_cut: int = 1800
    desc_micro_cut: int = 300
    tta: bool = False

cfg = Config()

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


def get_dist_and_sim(dict1, dict2):
    dist, sim = [], []
    try:
        dict1, dict2 = eval(dict1), eval(dict2)
        dict_keys = set(dict1.keys()) & set(dict2.keys())
    except:
        return dist, sim

    for key in dict_keys:
        val1 = dict1.get(key)
        val2 = dict2.get(key)
        if val1 != val2:
            dist.append(key)
        if val1 == val2:
            sim.append(key)
    return dist, sim

def preprocess_data(text_and_bert: pd.DataFrame, attributes: pd.DataFrame, test_pairs):
    text_and_bert['description'] = text_and_bert['description'].fillna('no desc')
    attributes = attributes[['categories', 'characteristic_attributes_mapping']]
    attributes['category_level_2'] = attributes['categories'].progress_apply(lambda x: eval(x)['2'])
    data = pd.concat([text_and_bert, attributes], axis=1)
    data['description'] = data['description'].progress_apply(remove_html_tags_and_emoji)
    data['name'] = data['name'].progress_apply(remove_html_tags_and_emoji)
    data['category_level_2'] = data['category_level_2'].progress_apply(lambda x: x.lower())

    test_pairs.rename(
        columns={
            'variantid1': 'variantid_1',
            'variantid2': 'variantid_2'
        }, inplace=True
    )

    test_df = test_pairs.merge(
        data.add_suffix('_1'),
        on='variantid_1'
    ).merge(
        data.add_suffix('_2'),
        on='variantid_2'
    )

    test_df['category_level_2'] = test_df['category_level_2_1']
    test_df.drop(columns=['category_level_2_1', 'category_level_2_2'], axis=1, inplace=True)

    dataset = []
    for i in tqdm(range(len(test_df))):
        row = test_df.iloc[i]
        target = -1
        category = row.category_level_2
        name1 = row.name_1
        name2 = row.name_2
        desc1 = row.description_1
        desc2 = row.description_2
        res_dist, res_similar = get_dist_and_sim(
            row.characteristic_attributes_mapping_1,
            row.characteristic_attributes_mapping_2
        )
        dataset.append(
            (category,
             name1,
             name2,
             desc1,
             desc2,
             ', '.join(res_dist),
             ', '.join(res_similar),
             target)
        )

    return dataset


def bert_inference_pipeline(text_and_bert: pd.DataFrame, attributes: pd.DataFrame, test_pairs: pd.DataFrame):
    dataset = preprocess_data(text_and_bert, attributes, test_pairs)

    tokenizer = BertTokenizer.from_pretrained(cfg.rubert_model, local_files_only=True)
    model_attr = BertForSequenceClassification.from_pretrained(cfg.rubert_model, num_labels=cfg.num_labels, local_files_only=True)
    model_desc = BertForSequenceClassification.from_pretrained(cfg.rubert_model, num_labels=cfg.num_labels, local_files_only=True)

    model_attr.load_state_dict(
        torch.load(
            cfg.model_attr,
            map_location=torch.device('cpu')
        )
    )

    model_desc.load_state_dict(
        torch.load(
            cfg.model_desc,
            map_location=torch.device('cpu')
        )
    )

    tokenizer_multi = AutoTokenizer.from_pretrained(cfg.distil_model, local_files_only=True)
    model_attr_multi = AutoModelForSequenceClassification.from_pretrained(cfg.distil_model, num_labels=cfg.num_labels, local_files_only=True)

    model_attr_multi.load_state_dict(
        torch.load(
            cfg.model_attr_multi,
            map_location=torch.device('cpu')
        )
    )

    eval_df = []
    for t in tqdm(range(len(dataset))):
        category, name1, name2, desc1, desc2, dist, sim, target = dataset[t]

        # ATTR
        s_attr_1 = category + '[SEP]' + name1 + '[SEP]' + name2 + '[SEP]' + dist
        tks_attr_1 = tokenizer.encode_plus(
            s_attr_1[:cfg.attr_cut], 
            max_length=cfg.max_len_attr, 
            pad_to_max_length=False, 
            return_attention_mask=True, 
            return_tensors='pt', 
            truncation=True
        )
        # DESC
        name_left = name1 + '[SEP]' + name2
        desc_start_left = desc1[:cfg.desc_micro_cut] + '[SEP]' + desc2[:cfg.desc_micro_cut]
        desc_end_left = desc1[-cfg.desc_micro_cut:] + '[SEP]' + desc2[-cfg.desc_micro_cut:] 
        s_desc_1 = category + '[SEP]' + 'Названия: ' + name_left + '[SEP]' + 'Описания: ' + desc_start_left + '[SEP]' + desc_end_left
        tks_desc_1 = tokenizer.encode_plus(
            s_desc_1[:cfg.desc_cut], 
            max_length=cfg.max_len_desc, 
            pad_to_max_length=False, 
            return_attention_mask=True, 
            return_tensors='pt', 
            truncation=True
        )
        # ATTR MULTI
        tks_attr_multi_1 = tokenizer_multi.encode_plus(
            s_attr_1[:cfg.attr_cut_multi], 
            max_length=cfg.max_len_attr_multi, 
            pad_to_max_length=False, 
            return_attention_mask=True, 
            return_tensors='pt', 
            truncation=True
        )

        if cfg.tta:
            # ATTR
            s_attr_2 = category + '[SEP]' + name2 + '[SEP]' + name1 + '[SEP]' + dist
            tks_attr_2 = tokenizer.encode_plus(
                s_attr_2[:cfg.attr_cut], 
                max_length=cfg.max_len_attr, 
                pad_to_max_length=False, 
                return_attention_mask=True, 
                return_tensors='pt', 
                truncation=True
            )
            # DESC
            name_right = name2 + '[SEP]' + name1
            desc_start_right = desc2[:cfg.desc_micro_cut] + '[SEP]' + desc1[:cfg.desc_micro_cut]
            desc_end_right = desc2[-cfg.desc_micro_cut:] + '[SEP]' + desc1[-cfg.desc_micro_cut:] 
            s_desc_2 = category + '[SEP]' + 'Названия: ' + name_right + '[SEP]' + 'Описания: ' + desc_start_right + '[SEP]' + desc_end_right
            tks_desc_2 = tokenizer.encode_plus(
                s_desc_2[:cfg.desc_cut], 
                max_length=cfg.max_len_desc, 
                pad_to_max_length=False, 
                return_attention_mask=True, 
                return_tensors='pt', 
                truncation=True
            )

            # ATTR MULTI
            tks_attr_multi_2 = tokenizer_multi.encode_plus(
                s_attr_2[:cfg.attr_cut_multi], 
                max_length=cfg.max_len_attr_multi, 
                pad_to_max_length=False, 
                return_attention_mask=True, 
                return_tensors='pt', 
                truncation=True
            )

        with torch.no_grad():
            score_attr_1 = model_attr(
                tks_attr_1['input_ids'], 
                attention_mask=tks_attr_1['attention_mask'],
                token_type_ids=tks_attr_1['token_type_ids']
            ).logits[0][1].item()
            
            score_desc_1 = model_desc(
                tks_desc_1['input_ids'], 
                attention_mask=tks_desc_1['attention_mask'],
                token_type_ids=tks_desc_1['token_type_ids']
            ).logits[0][1].item()

            score_attr_multi_1 = model_attr_multi(
                tks_attr_multi_1['input_ids'], 
                attention_mask=tks_attr_multi_1['attention_mask']
            ).logits[0][1].item()

            if cfg.tta:
                score_attr_2 = model_attr(
                    tks_attr_2['input_ids'], 
                    attention_mask=tks_attr_2['attention_mask'],
                    token_type_ids=tks_attr_2['token_type_ids']
                ).logits[0][1].item()
                
                score_desc_2 = model_desc(
                    tks_desc_2['input_ids'], 
                    attention_mask=tks_desc_2['attention_mask'],
                    token_type_ids=tks_desc_2['token_type_ids']
                ).logits[0][1].item()

                score_attr_multi_2 = model_attr_multi(
                    tks_attr_multi_2['input_ids'], 
                    attention_mask=tks_attr_multi_2['attention_mask']
                ).logits[0][1].item()

                score_attr = (score_attr_1 + score_attr_2) / 2
                score_desc = (score_desc_1 + score_desc_2) / 2
                score_attr_multi = (score_attr_multi_1 + score_attr_multi_2) / 2
            else:
                score_attr = score_attr_1
                score_desc = score_desc_1
                score_attr_multi = score_attr_multi_1

            eval_df.append(
                (test_pairs.iloc[t].variantid_1, 
                test_pairs.iloc[t].variantid_2, 
                score_attr, score_desc, score_attr_multi)
            )
            
    eval_df = pd.DataFrame(eval_df)
    eval_df.columns = ['variantid_1', 'variantid_2', 'name_attr_bert_oof', 'name_desc_bert_oof', 'multi_attr_bert_oof']

    return eval_df
