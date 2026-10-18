import numpy as np
import pandas as pd
import os
import random
import copy
import yaml
import gzip
import re
import html
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm,trange

def build_map(df,col_name):
    """
        Encodes a categorical column in a DataFrame by mapping unique values to integers
        and returns a decoding dictionary.

        :param df: The input DataFrame.
        :param col_name: The name of the column to be encoded.
        :return: dict: A dictionary mapping encoded integers back to original values.
    """
    key = df[col_name].unique().tolist()
    encode = dict(zip(key, range(len(key))))
    decode = dict(zip(range(len(key)),key))
    df[col_name] = df[col_name].apply(lambda x: encode[x])
    return decode

def build_item_map(df,col_name, history_name):
    """
        Encodes a categorical column in a DataFrame by mapping unique values with historical records to integers
        and returns a decoding dictionary.

        :param df: The input DataFrame.
        :param col_name: The name of the column to be encoded.
        :return: dict: A dictionary mapping encoded integers back to original values.
    """


    item_set = []
    for i in range(len(df)):
        item_set.append(df[col_name][i])
        for h_i in df[history_name][i]:
            item_set.append(h_i)

    # Keep the item-id space deterministic so it stays aligned with external
    # SID tables built by GRID, which use a fixed sorted item mapping.
    key = sorted(set(item_set))
    encode = dict(zip(key, range(len(key))))
    decode = dict(zip(range(len(key)),key))
    df[col_name] = df[col_name].apply(lambda x: encode[x])
    df[history_name] = df[history_name].apply(lambda x: [encode[i] for i in x])
    return decode, len(key)

from fairdiverse.recommendation.dataset_loaders import (
    DatasetLoader, RecBoleLoader, McAuleyRawLoader, _detect_price_col
)


def Process(dataset, config=None):
    """
        Pre-process dataset, split training-validate-test set into ~/processed_data path

        :param dataset: str: The utilized dataset.
            Note that please download it into ~/dataset path

        :param config: The custom config files.
    """

    # current_dir = os.getcwd()
    # print(f"Current directory: {current_dir}")  # Debug line
    #with open(os.path.join(current_dir, "fairdiverse", "recommendation","properties", "dataset.yaml"), "r")
    with open(os.path.join("fairdiverse", "recommendation","properties", "dataset.yaml"), "r") as file:
        process_config = yaml.safe_load(file)


    with open(os.path.join("fairdiverse", "recommendation","properties", "dataset", "{}.yaml".format(dataset)), "r") as file:
        dataset_config = yaml.safe_load(file)

    process_config.update(dataset_config)

    if config:
        process_config.update(config)

    print("process config:")
    print(process_config)


    if os.path.exists(os.path.join("recommendation", "processed_dataset", str(dataset))) and process_config['reprocess'] == False:
        return "do not need to process"

    print("start to process data...")

    # Select loader based on config:
    # - McAuleyRawLoader when 'reviews_path' is set (reads raw McAuley JSON directly)
    # - RecBoleLoader otherwise (reads pre-converted RecBole TSV files)
    if process_config.get('reviews_path'):
        print("Using McAuleyRawLoader (reading directly from McAuley JSON/CSV files)")
        loader = McAuleyRawLoader(process_config)
    else:
        print("Using RecBoleLoader (reading from RecBole-format files)")
        loader = RecBoleLoader(process_config, dataset)

    frames         = loader.load_interactions()
    provider_frames = loader.load_items()
    mcauley_meta   = loader.load_metadata()

    # Detect available enrichment columns
    available_cols = provider_frames.columns.tolist()
    categories_col = "categories:token_seq"
    brand_col      = "brand:token"
    has_categories = categories_col in available_cols
    has_brand      = brand_col in available_cols
    price_col, has_price = _detect_price_col(available_cols)

    print(len(provider_frames))

    frames = copy.copy(frames.sample(frac=process_config['sample_size'], random_state=42, axis=0, replace=False).reset_index(drop=True))


    #print("start to pre-process data...")

    uid_field,iid_field,label_field, time_field, text_field = \
        [process_config["user_id"],process_config["item_id"],process_config["label_id"], process_config['timestamp'], process_config["text_id"]]
    def clean_categories(raw):
        if not isinstance(raw, str) or not raw.strip():
            return ""
        parts = [p.strip().strip("'\"") for p in raw.split(",")]
        parts = [p for p in parts if p]
        return " > ".join(parts)

    if has_price or has_brand or mcauley_meta:
        # Build rich text from available RecBole fields + optional external metadata:
        # "Title: X, Category: X, Description: X, Price: $X"
        def build_rich_text(asin, recbole_title, recbole_cats=None, recbole_price=None, recbole_brand=None):
            meta = mcauley_meta.get(str(asin), {}) if mcauley_meta else {}
            parts = []
            
            # Extract Title
            title = str(meta.get('title', '') or recbole_title or '').strip()
            if title:
                title = re.sub(r'<[^>]+>', ' ', title)
                title = re.sub(r'&[a-zA-Z#][a-zA-Z0-9]+;', lambda m: html.unescape(m.group()), title)
                title = ' '.join(title.split())
                parts.append(f"Title: {title}")
            
            # Extract and combine Category (category list + main_cat)
            main_cat = str(meta.get('main_cat', '') or '').strip()
            cats = meta.get('category', []) or []
            if not isinstance(cats, list):
                cats = []
            cats = [c for c in cats if c]
            # Combine: category list first, then main_cat (if not already in list)
            cat_parts = list(cats)  # Start with category list
            if main_cat and main_cat not in cat_parts:
                cat_parts.append(main_cat)
            # Fallback to RecBole categories if no metadata categories
            if not cat_parts and recbole_cats:
                fallback = clean_categories(str(recbole_cats))
                if fallback:
                    cat_parts = [fallback]
            if cat_parts:
                parts.append(f"Category: {' > '.join(cat_parts)}")
            
            # Extract Description
            desc = meta.get('description', []) or []
            if isinstance(desc, list):
                desc_text = ' '.join(str(d).strip() for d in desc if str(d).strip())
            else:
                desc_text = str(desc).strip()
            if desc_text:
                # Strip HTML tags (including <img src="..."> URLs embedded in attributes)
                desc_text = re.sub(r'<[^>]+>', ' ', desc_text)
                # Remove any remaining bare URLs (http/https/www)
                desc_text = re.sub(r'https?://\S+|www\.\S+', '', desc_text)
                # Remove HTML entities (e.g. &amp; &quot; &#39;)
                desc_text = re.sub(r'&[a-zA-Z#][a-zA-Z0-9]+;', ' ', desc_text)
                desc_text = ' '.join(desc_text.split())
                if desc_text:
                    parts.append(f"Description: {desc_text[:500]}")
            
            # Extract Price
            price = str(meta.get('price', '') or '').strip()
            if not price and recbole_price is not None:
                price_str = str(recbole_price).strip()
                if price_str not in ('', 'nan', 'NaN'):
                    # price:token already has "$"; price:float is a plain number
                    price = price_str if price_str.startswith('$') else f"${price_str}"
            if price:
                parts.append(f"Price: {price}")
            
            return ', '.join(parts)

        id2text = {}
        for _, row in provider_frames.iterrows():
            asin = row[iid_field]
            cats = row[categories_col] if has_categories else None
            price = row[price_col] if has_price else None
            brand = row[brand_col] if has_brand else None
            id2text[asin] = build_rich_text(asin, row[text_field], cats, price, brand)
    elif has_categories:
        provider_frames["combined_text"] = (
            provider_frames[text_field].fillna("")
            + " | "
            + provider_frames[categories_col].fillna("").apply(clean_categories)
        )
        id2text = dict(zip(provider_frames[iid_field], provider_frames["combined_text"]))
    else:
        id2text = dict(zip(provider_frames[iid_field], provider_frames[text_field]))


    provider_field = process_config["group_id"]

    frames.drop_duplicates(subset=[iid_field,uid_field],keep='first',inplace=True)
    #print(uid_field,iid_field,label_field,time_field)
    item_num, user_num = len(frames[iid_field].unique()),len(frames[uid_field].unique())
    group_num = len(provider_frames[provider_field].unique())
    print("origin:----------item number: %d user number:%d group_num:%d total interactions:%d"%(item_num,user_num,group_num, len(frames)))



    print("start to merge data.....")

    print("processing item val")
    frames[label_field] = frames[label_field].apply(lambda x: 1 if x>=process_config['label_threshold'] else 0)

    frames.rename(columns={label_field:"label:float"},inplace=True)
    label_field = "label:float"
    frames = frames.merge(provider_frames,on=iid_field,how='inner')
    # Fill NaN brands so they are not silently dropped by groupby/isin in the filtering loop
    frames[provider_field] = frames[provider_field].fillna('no_brand')
    #print(len(frames))
    #exit(0)

    for i in range(3):
        itemLen = frames.groupby(iid_field).size()  # groupby itemID and get size of each item
        remain_items =  (itemLen[itemLen >= process_config['item_val']].index).values
        frames = frames[frames[iid_field].isin(remain_items)].reset_index(drop=True)
        #print(len(frames))
        #print(itemLen)
        userLen = frames.groupby(uid_field).size()
        remain_users =  (userLen[userLen >= process_config['user_val']].index).values
        frames = frames[frames[uid_field].isin(remain_users)].reset_index(drop=True)
        #print(len(frames))
        pidLen = frames.groupby(provider_field)[iid_field].unique().apply(lambda x: len(x))
        remain_providers =  (pidLen[pidLen >= process_config['group_val']].index).values
        frames = frames[frames[provider_field].isin(remain_providers)].reset_index(drop=True)

        #print(len(frames))
        #exit(0)

    frames = frames.sort_values(by=[uid_field, time_field]).reset_index(drop=True)

    # Build history_behaviors per user explicitly (avoids pandas version issues with groupby.apply)
    #
    # Standalone GRID preprocessing keeps the full chronological sequence and
    # lets the training/inference pipeline trim or pad later. To keep
    # FairDiverse's leave-one-out split while matching GRID's sequence
    # construction more closely, disable the preprocessing-time history cap for
    # GRID runs.
    preserve_full_history = process_config.get('model') == 'GRID'
    result_dfs = []
    for uid, sub_df in frames.groupby(uid_field, sort=False):
        clicked_items = []
        previous_items = []
        for _, row in sub_df.iterrows():
            h = None if preserve_full_history else process_config.get('history_length')
            previous_items.append(clicked_items[-h:] if h else clicked_items[:])
            if row[label_field] == 1:
                clicked_items.append(row[iid_field])
        sub_df = sub_df.copy()
        sub_df['history_behaviors'] = previous_items
        result_dfs.append(sub_df)
    frames = pd.concat(result_dfs, ignore_index=True)

    # history_length is a CAP only (take last N items) — no equality filter.
    # All users are kept, matching GRID's min_sequence_length=1.
    if preserve_full_history:
        print("Preserving full chronological histories during preprocessing for GRID alignment.")
    history_field = 'history_behaviors'
    process_config['history_field'] = 'history_behaviors'

    user_num, group_num = len(frames[uid_field].unique()), len(frames[provider_field].unique())





    min_item_size = 99999
    max_item_size = -1

    #build_map(frames,iid_field)
    iid_decode, item_num = build_item_map(frames, iid_field, history_field)
    _ = build_map(frames,uid_field)


    pid2iid = {}
    iid2pid = {}
    for i, (iid, pid) in enumerate(zip(frames[iid_field].values, frames[provider_field].values)):
        if pid not in pid2iid:
            pid2iid[pid] = []
        pid2iid[pid].append(iid)
        iid2pid[iid] = pid
    for key in pid2iid.keys():
        pid2iid[key] = list(set(pid2iid[key]))


    ###Since some group contains few items, we need to aggregate the some groups into one group
    ##SO WE NEED TO remap the provider id
    #index = 1
    remap_dict = {}
    for pid in frames[provider_field].unique():
        items = pid2iid[pid]
        if len(items) < process_config['group_aggregation_threshold']:
            for i in items:
                remap_dict[i] = -1
    iid2pid.update(remap_dict)
    #print(iid2pid)

    frames['remap_pid'] = frames[iid_field].map(iid2pid)
    provider_field = 'remap_pid'
    build_map(frames, provider_field)

    pid2iid = {}
    iid2pid = {}
    for i, (iid, pid) in enumerate(zip(frames[iid_field].values, frames[provider_field].values)):
        iid = str(iid)
        pid = str(pid)
        if pid not in pid2iid:
            pid2iid[pid] = []
        pid2iid[pid].append(iid)
        iid2pid[iid] = pid
    for key in pid2iid.keys():
        pid2iid[key] = str(list(set(pid2iid[key])))

    # item_num, user_num, group_num = len(frames[iid_field].unique()), len(frames[uid_field].unique()), len(
    #     frames[provider_field].unique())
    # print("after process.... item number: %d user number:%d group number:%d total interactions:%d" % (
    # item_num, user_num, group_num, len(frames)))
    group_num = len(frames[provider_field].unique())

    process_config['user_num'] = user_num
    process_config['item_num'] = item_num
    process_config['group_num'] = group_num
    process_config['label_id'] = label_field

    print("item number: %d user number:%d group number:%d total interactions:%d" % (
        item_num, user_num, group_num, len(frames)))

    # Save item_num, user_num, group_num to a config file
    stats_file = os.path.join("recommendation", "processed_dataset", str(dataset), "dataset_stats.json")
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump({
            'item_num': int(item_num),
            'user_num': int(user_num), 
            'group_num': int(group_num)
        }, f)
    print(f"Saved dataset statistics to {stats_file}")

    print("start to sample and split data....")
    frames = frames.sample(frac=1, random_state=42).reset_index(drop=True)
    #frames = frames.reset_index(drop=True)


    #print(frames.head())
    train_items = frames[iid_field].unique()
    train_users = frames[uid_field].unique()
    print("checking.....")


    if np.sum(train_users>=len(train_users)) > 0:
        print("ERROR in user id:",np.sum(train_users>=len(train_users)))
        exit(0)

    print("min user interactions:",np.min(frames.groupby(uid_field).size()))
    print("min item interactions:",np.min(frames.groupby(iid_field).size()))



    final_frame = frames[[uid_field, iid_field, history_field, label_field, provider_field, time_field]]
    final_frame.rename(columns={
        'remap_pid': process_config['group_id']
    }, inplace=True)

    eval_split_method = process_config.get('eval_split_method', 'time_split')
    print(f"Using eval split method: {eval_split_method}")

    if eval_split_method == 'leave_one_out':
        # Per-user leave-one-out: last positive per user → test,
        # second-to-last → val, everything else → train.
        # This matches the evaluation protocol used in TIGER/GRID papers and
        # ensures each user has exactly 1 test positive (unbiased NDCG).
        final_frame = final_frame.sort_values(by=[uid_field, time_field]).reset_index(drop=True)
        pos_df = final_frame[final_frame[label_field] == 1]
        test_indices, val_indices = [], []
        for uid, user_pos in pos_df.groupby(uid_field):
            sorted_pos = user_pos.sort_values(time_field)
            n = len(sorted_pos)
            if n >= 2:
                test_indices.append(sorted_pos.index[-1])
                val_indices.append(sorted_pos.index[-2])
            elif n == 1:
                # Only 1 positive: use as val; user will not appear in test
                val_indices.append(sorted_pos.index[-1])
        test_mask = final_frame.index.isin(test_indices)
        val_mask = final_frame.index.isin(val_indices)
        test = final_frame[test_mask].reset_index(drop=True)
        val = final_frame[val_mask].reset_index(drop=True)
        train = final_frame[~test_mask & ~val_mask].reset_index(drop=True)
        print(f"leave-one-out split: train={len(train)} rows, val={len(val)} users, test={len(test)} users")
    else:
        # Original global time-split (80/10/10 by interaction count)
        final_frame.sort_values(by=time_field, inplace=True)
        train_size = int(len(final_frame) * (1 - process_config['valid_ratio'] - process_config['test_ratio']))
        val_size = int(len(final_frame) * process_config['valid_ratio'])
        train = final_frame.iloc[:train_size]
        val = final_frame.iloc[train_size:train_size + val_size]
        test = final_frame.iloc[train_size + val_size:]

    print(test.head())
    #split train-val-test accoring to the time


    # This is for extracting the positive items for every user to make it easy to conduct negative samples for each users
    user2pos_itemset = {}
    for i in range(len(final_frame)):
        uid = final_frame[uid_field][i]
        iid = final_frame[iid_field][i]
        label = final_frame[label_field][i]
        if uid not in user2pos_itemset.keys():
            user2pos_itemset[uid] = []
        if label == 1:
            user2pos_itemset[uid].append(iid)


    dir = os.path.join("recommendation","processed_dataset",dataset)
    if not os.path.exists(dir):
        os.makedirs(dir)



    #columns = [process_config['user_id'], process_config['item_id'], process_config['label_id'], process_config['group_id']]


    test.to_csv(os.path.join(dir,dataset+".test.CTR"), index=False,sep='\t')
    val.to_csv(os.path.join(dir,dataset+".valid.CTR"), index=False,sep='\t')
    train.to_csv(os.path.join(dir,dataset+".train"), index=False,sep='\t')
    from .user_fairness import write_activity, fingerprint
    activity_counts = dict.fromkeys(final_frame[uid_field].unique().tolist(), 0)
    activity_counts.update(train[train[label_field] == 1].groupby(uid_field).size().to_dict())
    write_activity(activity_counts, dir, {
        "kind": "positive_training_rows", "dataset": dataset,
        "training": fingerprint(os.path.join(dir, dataset + ".train")),
        "user_id_mapping": "processed dataset IDs", "split_method": eval_split_method,
    })

    #iid2pid
    def construct_ranking_data(df):

        sort_df = df.sort_values(by=[uid_field, time_field]).reset_index(drop=True)
        #construct_dict = {"user_id":[], "items":[], "pos_length":[] }

        construct_dict = {}
        user_behaviors = {}

        for i in trange(len(sort_df)):

            user, item, label = sort_df[uid_field].values[i], sort_df[iid_field].values[i], sort_df[label_field].values[i]
            history = sort_df[history_field].values[i]
            ##find the behavior sequences in the training dataset
            if label == 1:
                if user not in construct_dict.keys():
                    construct_dict[user] = [item]
                    # filtered_df = train[(train[uid_field] == user) & (train[label_field] == 1)]
                    # print(filtered_df.head())
                    # filtered_df['time_diff'] = abs(filtered_df[time_field] - time)  # find the most close behavior
                    # closest_row = filtered_df.loc[filtered_df['time_diff'].idxmin()]
                    # behavior = closest_row["history_behaviors"][-(process_config['history_length']-1):]
                    # behavior.append(closest_row[iid_field])
                    user_behaviors[user] = history

                else:
                    construct_dict[user].append(item)

        df_dict = {"user_id":[], "history_behaviors":[], "items":[], "pos_length":[] }
        max_len = 0
        for u in tqdm(construct_dict.keys()):
            pos_length = len(construct_dict[u])
            itemset = construct_dict[u]
            if pos_length > max_len:
                max_len = pos_length

            num_negs_needed = process_config['sample_num'] - pos_length
            pos_set = set(user2pos_itemset[u]) | set(itemset)
            available_negs = item_num - len(pos_set)
            num_negs_needed = min(num_negs_needed, available_negs)

            if num_negs_needed > 0:
                if num_negs_needed >= available_negs * 0.5:
                    # Full-catalog (or near-full): enumerate to avoid slow rejection sampling
                    all_negs = [i for i in range(item_num) if i not in pos_set]
                    random.shuffle(all_negs)
                    itemset.extend(all_negs[:num_negs_needed])
                else:
                    for r in range(num_negs_needed):
                        neg_id = random.randint(0, item_num - 1)
                        while neg_id in pos_set:
                            neg_id = random.randint(0, item_num - 1)
                        itemset.append(neg_id)
                        pos_set.add(neg_id)

            # Pad itemset to sample_num so all rows have equal length for torch.stack
            if len(itemset) < process_config['sample_num']:
                itemset = itemset + [0] * (process_config['sample_num'] - len(itemset))

            df_dict["user_id"].append(u)
            df_dict["history_behaviors"].append(user_behaviors[u])
            df_dict["items"].append(itemset)
            df_dict["pos_length"].append(pos_length)

        df_reconstruct = pd.DataFrame(df_dict)
        print(f"max_len:{max_len}")
        return df_reconstruct

    print("start to construct ranking and retrieval test dataset...")
    val_ranking = construct_ranking_data(val)
    test_ranking = construct_ranking_data(test)

    val_ranking["items"] = val_ranking["items"].apply(lambda lst: [int(x) for x in lst])
    test_ranking["items"] = test_ranking["items"].apply(lambda lst: [int(x) for x in lst])

    val_ranking.to_csv(os.path.join(dir,dataset+".valid.ranking"), index=False,sep='\t')
    test_ranking.to_csv(os.path.join(dir,dataset+".test.ranking"), index=False,sep='\t')

    id2text_update = {}
    for i in range(item_num):
        ori_i = iid_decode[i]
        text = id2text[ori_i]
        id2text_update[i] = text

    with open(os.path.join(dir, "process_config.yaml"), "w") as file:
        yaml.dump(process_config, file)

    #print(iid2pid)
    with open(os.path.join(dir, "iid2pid.json"), "w") as file:
        json.dump(iid2pid, file)

    with open(os.path.join(dir, "iid2text.json"), "w") as file:
        json.dump(id2text_update, file)

    # Preserve the original item identifier for parity checks against GRID's
    # item_id_mapping.json. Both pipelines use sorted item IDs, but recording
    # the mapping lets us prove alignment instead of assuming it.
    with open(os.path.join(dir, "iid2item.json"), "w") as file:
        json.dump({str(i): iid_decode[i] for i in range(item_num)}, file)


    return f"process complete! The file and config are stored in {dir}"
