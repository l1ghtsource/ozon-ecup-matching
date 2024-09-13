import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
import gc
import re
from tqdm import tqdm

tqdm.pandas()


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


def pr_auc_macro(df: pd.DataFrame) -> float:
    y_true = df["target"]
    y_pred = df["scores"]
    categories = df["categories"]
    
    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    for i, category in enumerate(unique_cats):
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        if sum(y_true_cat) == 0:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        
        precision, recall, _ = precision_recall_curve(y_true_cat, y_pred_cat)
        
        try:
            pr_auc = auc(recall, precision)
            if not np.isnan(pr_auc):
                pr_aucs.append(pr_auc)
        except ValueError:
            pr_aucs.append(0)

    return np.average(pr_aucs)


def prepare_dataset():
    text_and_bert = pd.read_parquet('./data/train/text_and_bert.parquet', engine='pyarrow')
    train_pairs = pd.read_parquet('./data/train/train.parquet', engine='pyarrow')
    attrs = pd.read_parquet('./data/train/attributes.parquet', columns=['categories', 'characteristic_attributes_mapping'], engine='pyarrow')
    
    text_and_bert['description'] = text_and_bert['description'].fillna('no desc')
    attrs['category_level_2'] = attrs['categories'].progress_apply(lambda x: eval(x)['2'])
    data = pd.concat([text_and_bert, attrs], axis=1)

    del text_and_bert, attrs
    gc.collect()

    data['description'] = data['description'].progress_apply(remove_html_tags_and_emoji)
    data['name'] = data['name'].progress_apply(remove_html_tags_and_emoji)
    data['category_level_2'] = data['category_level_2'].progress_apply(lambda x: x.lower())

    train_pairs.rename(
        columns={
            'variantid1': 'variantid_1',
            'variantid2': 'variantid_2'
        }, inplace=True
    )

    train_df = train_pairs.merge(
        data.add_suffix('_1'), 
        on='variantid_1'
    ).merge(
        data.add_suffix('_2'), 
        on='variantid_2'
    )

    train_df['category_level_2'] = train_df['category_level_2_1']
    train_df.drop(columns=['category_level_2_1', 'category_level_2_2'], axis=1, inplace=True)

    dataset = []
    for i in tqdm(range(len(train_df))):
        row = train_df.iloc[i]
        target = row.target
        category = row.category_level_2
        name1 = row.name_1
        name2 = row.name_2
        desc1 = row.description_1
        desc2 = row.description_2
        res_dist, res_similar = get_dist_and_sim(
            row.characteristic_attributes_mapping_1,
            row.characteristic_attributes_mapping_2
        )
        dataset.append((
            category, 
            name1, 
            name2,
            desc1, 
            desc2, 
            ', '.join(res_dist), 
            ', '.join(res_similar), 
            target
        ))
        
    return dataset


def full_train_attr(model_name, dataset, max_len, cut, name, epochs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()

    batch_size = 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    total_steps = (1 + len(dataset) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=total_steps)

    train_losses = []
    for ep in range(epochs):
        np.random.shuffle(dataset)
        optimizer.zero_grad()
        losses = []
        pbar = tqdm(range(len(dataset)), desc=f'Epoch {ep} Loss 0.000', total=len(dataset))
        for t in pbar:
            category, name1, name2, desc1, desc2, dist, sim, target = dataset[t]
            s = category + '[SEP]' + name1 + '[SEP]' + name2 + '[SEP]' + dist
            tks = tokenizer.encode_plus(
                s[:cut], 
                max_length=max_len, 
                pad_to_max_length=False, 
                return_attention_mask=True, 
                return_tensors='pt', 
                truncation=True
            )
            out = model(
                tks['input_ids'].cuda(), 
                attention_mask=tks['attention_mask'].cuda(),
                token_type_ids=tks['token_type_ids'].cuda(),
                labels=torch.tensor([[1.0 - target, target]]).float().cuda()
            )
            
            losses.append(out.loss)
            
            if (t + 1) % batch_size == 0:
                loss = sum(losses) / batch_size
                loss.backward()
                losses = []
                train_losses.append(loss.item())
                optimizer.step() 
                optimizer.zero_grad()
                scheduler.step()
                pbar.set_description(f'Epoch {ep} Loss {loss.item():.3f}')
        
        if len(losses) > 0:
            loss = sum(losses) / batch_size
            loss.backward()
            losses = []
            train_losses.append(loss.item())
            optimizer.step() 
            optimizer.zero_grad()
            scheduler.step()
            pbar.set_description(f'Epoch {ep} Loss {loss.item():.3f}')

    torch.save(model.state_dict(), f'./models/BERT/{name}.pth')


def get_oof_attr(model_name, dataset, max_len, cut, name, epochs):
    kf = KFold(n_splits=5, shuffle=True, random_state=1488)
    ds_indexes = np.arange(len(dataset))

    batch_size = 32
    ifold = 0 
    oof = np.zeros(len(dataset))

    for tr, va in kf.split(ds_indexes):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        total_steps = (1 + len(tr) // batch_size) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=0, 
                                                    num_training_steps=total_steps)
            
        i = 0
        train_losses = []
        for ep in range(epochs):
            np.random.shuffle(tr)
            optimizer.zero_grad()
            losses = []
            pbar = tqdm(tr, desc=f'Fold {ifold} Epoch {ep} Loss 0.000', total=len(tr))
            for t in pbar:
                category, name1, name2, desc1, desc2, dist, sim, target = dataset[t]
                s = category + '[SEP]' + name1 + '[SEP]' + name2 + '[SEP]' + dist
                tks = tokenizer.encode_plus(
                    s[:cut], 
                    max_length=max_len, 
                    pad_to_max_length=False, 
                    return_attention_mask=True, 
                    return_tensors='pt', 
                    truncation=True
                )
                out = model(
                    tks['input_ids'].cuda(), 
                    attention_mask=tks['attention_mask'].cuda(),
                    token_type_ids=tks['token_type_ids'].cuda(),
                    labels = torch.tensor([[1.0 - target, target]]).float().cuda()
                )
                
                losses.append(out.loss)
                
                i += 1
                if i % batch_size == 0:
                    loss = sum(losses) / batch_size
                    loss.backward()
                    losses = []
                    train_losses.append(loss.item())
                    optimizer.step() 
                    optimizer.zero_grad()
                    scheduler.step()
                    pbar.set_description(f'Fold {ifold} Epoch {ep} Loss {loss.item():.3f}')
                    
            if len(losses) > 0:
                loss = sum(losses) / batch_size
                loss.backward()
                losses = []
                train_losses.append(loss.item())
                optimizer.step() 
                optimizer.zero_grad()
                scheduler.step()
                pbar.set_description(f'Fold {ifold} Epoch {ep} Loss {loss.item():.3f}')
                
            evaldf = []
            for t in va:
                category, name1, name2, desc1, desc2, dist, sim, target = dataset[t]
                s = category + '[SEP]' + name1 + '[SEP]' + name2 + '[SEP]' + dist
                tks = tokenizer.encode_plus(
                    s[:cut], 
                    max_length=max_len, 
                    pad_to_max_length=False, 
                    return_attention_mask=True, 
                    return_tensors='pt', 
                    truncation=True
                )

                with torch.no_grad():
                    score = model(
                        tks['input_ids'].cuda(), 
                        attention_mask=tks['attention_mask'].cuda(),
                        token_type_ids=tks['token_type_ids'].cuda()
                    ).logits[0][1].item()
                evaldf.append((target, score, category))
                
            evaldf = pd.DataFrame(evaldf)
            evaldf.columns = ["target", "scores", "categories"]
            
            m = pr_auc_macro(evaldf)
            m2 = roc_auc_score(evaldf.target.values, evaldf.scores.values)
            
            print('fold', ifold, 'epoch', ep, 'pr-auc', round(m, 3), 'roc-auc', round(m2, 3))
            torch.save(model.state_dict(), f'./models/BERT/{name}_{ifold}_{round(m, 3)}_{round(m2, 3)}.pth')
            oof[va] = evaldf.scores.values
            
        ifold += 1
        
    df_oof = pd.read_parquet('./data/train/train.parquet')
    df_oof[f'{name}_oof'] = oof
    df_oof.to_parquet(f'./data/train/{name}_oof.parquet')
    

def train_desc_full(model_name, dataset, max_len, cut, micro_cut, name, epochs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()

    batch_size = 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    total_steps = (1 + len(dataset) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=total_steps)

    train_losses = []
    for ep in range(epochs):
        np.random.shuffle(dataset)
        optimizer.zero_grad()
        losses = []
        pbar = tqdm(range(len(dataset)), desc=f'Epoch {ep} Loss 0.000', total=len(dataset))
        for t in pbar:
            category, name1, name2, desc1, desc2, dist, sim, target = dataset[t]
            name = name1 + '[SEP]' + name2
            desc_start = desc1[:micro_cut] + '[SEP]' + desc2[:micro_cut]
            desc_end = desc1[-micro_cut:] + '[SEP]' + desc2[-micro_cut:] 
            s = category + '[SEP]' + 'Названия: ' + name + '[SEP]' + 'Описания: ' + desc_start + '[SEP]' + desc_end
            tks = tokenizer.encode_plus(
                s[:cut], 
                max_length=max_len, 
                pad_to_max_length=False, 
                return_attention_mask=True, 
                return_tensors='pt', 
                truncation=True
            )
            out = model(
                tks['input_ids'].cuda(), 
                attention_mask=tks['attention_mask'].cuda(),
                token_type_ids=tks['token_type_ids'].cuda(),
                labels=torch.tensor([[1.0 - target, target]]).float().cuda()
            )
            
            losses.append(out.loss)
            
            if (t + 1) % batch_size == 0:
                loss = sum(losses) / batch_size
                loss.backward()
                losses = []
                train_losses.append(loss.item())
                optimizer.step() 
                optimizer.zero_grad()
                scheduler.step()
                pbar.set_description(f'Epoch {ep} Loss {loss.item():.3f}')
        
        if len(losses) > 0:
            loss = sum(losses) / batch_size
            loss.backward()
            losses = []
            train_losses.append(loss.item())
            optimizer.step() 
            optimizer.zero_grad()
            scheduler.step()
            pbar.set_description(f'Epoch {ep} Loss {loss.item():.3f}')

    torch.save(model.state_dict(), f'./models/BERT/{name}.pth')
    

def get_oof_desc(model_name, dataset, max_len, cut, micro_cut, name, epochs):
    kf = KFold(n_splits=5, shuffle=True, random_state=1488)
    ds_indexes = np.arange(len(dataset))

    batch_size = 32
    ifold = 0 
    oof = np.zeros(len(dataset))

    for tr, va in kf.split(ds_indexes):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        total_steps = (1 + len(tr) // batch_size) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=0, 
                                                    num_training_steps=total_steps)
    
        i = 0
        train_losses = []
        for ep in range(epochs):
            np.random.shuffle(tr)
            optimizer.zero_grad()
            losses = []
            pbar = tqdm(tr, desc=f'Fold {ifold} Epoch {ep} Loss 0.000', total=len(tr))
            for t in pbar:
                category, name1, name2, desc1, desc2, dist, sim, target = dataset[t]
                name = name1 + '[SEP]' + name2
                desc_start = desc1[:micro_cut] + '[SEP]' + desc2[:micro_cut]
                desc_end = desc1[-micro_cut:] + '[SEP]' + desc2[-micro_cut:] 
                s = category + '[SEP]' + 'Названия: ' + name + '[SEP]' + 'Описания: ' + desc_start + '[SEP]' + desc_end
                tks = tokenizer.encode_plus(
                    s[:cut], 
                    max_length=max_len, 
                    pad_to_max_length=False, 
                    return_attention_mask=True, 
                    return_tensors='pt', 
                    truncation=True
                )
                out = model(
                    tks['input_ids'].cuda(), 
                    attention_mask=tks['attention_mask'].cuda(),
                    token_type_ids=tks['token_type_ids'].cuda(),
                    labels = torch.tensor([[1.0 - target, target]]).float().cuda()
                )
                
                losses.append(out.loss)
                
                i += 1
                if i % batch_size == 0:
                    loss = sum(losses) / batch_size
                    loss.backward()
                    losses = []
                    train_losses.append(loss.item())
                    optimizer.step() 
                    optimizer.zero_grad()
                    scheduler.step()
                    pbar.set_description(f'Fold {ifold} Epoch {ep} Loss {loss.item():.3f}')
                    
            if len(losses) > 0:
                loss = sum(losses) / batch_size
                loss.backward()
                losses = []
                train_losses.append(loss.item())
                optimizer.step() 
                optimizer.zero_grad()
                scheduler.step()
                pbar.set_description(f'Fold {ifold} Epoch {ep} Loss {loss.item():.3f}')
                
            evaldf = []
            for t in va:
                category, name1, name2, desc1, desc2, dist, sim, target = dataset[t]
                name = name1 + '[SEP]' + name2
                desc_start = desc1[:micro_cut] + '[SEP]' + desc2[:micro_cut]
                desc_end = desc1[-micro_cut:] + '[SEP]' + desc2[-micro_cut:] 
                s = category + '[SEP]' + 'Названия: ' + name + '[SEP]' + 'Описания: ' + desc_start + '[SEP]' + desc_end
                tks = tokenizer.encode_plus(
                    s[:cut], 
                    max_length=max_len, 
                    pad_to_max_length=False, 
                    return_attention_mask=True, 
                    return_tensors='pt', 
                    truncation=True
                )

                with torch.no_grad():
                    score = model(
                        tks['input_ids'].cuda(), 
                        attention_mask=tks['attention_mask'].cuda(),
                        token_type_ids=tks['token_type_ids'].cuda()
                    ).logits[0][1].item()
                evaldf.append((target, score, category))
                
            evaldf = pd.DataFrame(evaldf)
            evaldf.columns = ["target", "scores", "categories"]
            
            m = pr_auc_macro(evaldf)
            m2 = roc_auc_score(evaldf.target.values, evaldf.scores.values)
            
            print('fold', ifold, 'epoch', ep, 'pr-auc', round(m, 3), 'roc-auc', round(m2, 3))
            torch.save(model.state_dict(), f'./models/BERT/{name}_{ifold}_{round(m, 3)}_{round(m2, 3)}.pth')
            oof[va] = evaldf.scores.values
            
        ifold += 1
           
    df_oof = pd.read_parquet('./data/train/train.parquet') # TODO: путь
    df_oof[f'{name}_oof'] = oof
    df_oof.to_parquet(f'./data/train/{name}_oof.parquet')


def training_berts():
    print('Preprocessing data for BERTs...')
    dataset = prepare_dataset()
    
    print('Training Attributes Distilbert (FULL)...')
    full_train_attr(
        model_name='distilbert/distilbert-base-multilingual-cased',
        dataset=dataset,
        max_len=512,
        cut=1200,
        name='multi512_attr_bert_full_second_epoch',
        epochs=1
    )
    
    print('Training Attributes Distilbert (OOF)...')
    get_oof_attr(
        model_name='distilbert/distilbert-base-multilingual-cased',
        dataset=dataset,
        max_len=512,
        cut=1200,
        name='multi_attr_bert',
        epochs=1
    )
    
    print('Training Attributes RuBERT (FULL)...')
    full_train_attr(
        model_name='cointegrated/rubert-tiny2',
        dataset=dataset,
        max_len=768,
        cut=1600,
        name='3epoch_768_name_attr_bert_full',
        epochs=3
    )
    
    print('Training Attributes RuBERT (OOF)...')
    get_oof_attr(
        model_name='cointegrated/rubert-tiny2',
        dataset=dataset,
        max_len=768,
        cut=1600,
        name='name_attr_bert',
        epochs=3
    )

    print('Training Description RuBERT (FULL)...')
    train_desc_full(
        model_name='cointegrated/rubert-tiny2',
        dataset=dataset,
        max_len=1024,
        cut=1800,
        micro_cut=300,
        name='3epoch_1024_name_desc_bert_full',
        epochs=3
    )
    
    print('Training Description RuBERT (OOF)...')
    get_oof_desc(
        model_name='cointegrated/rubert-tiny2',
        dataset=dataset,
        max_len=1024,
        cut=1800,
        micro_cut=300,
        name='name_desc_bert',
        epochs=3
    )
