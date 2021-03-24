import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

import fire

from load_data import load_user_time_read
from preprocessing import PostIdEncoder
from tfidf import TFIDFGenerator
from config import Config
from utils import save_as_pickle, squeeze
 

start = Config.train_start
end = Config.train_end

def filter_read_by_time(history, start, end):
    history_filtered = list(filter(lambda x: (int(start) <= int(x[0].split('_')[0])) and (int(end) >= int(x[0].split('_')[-1])), history))
    return history_filtered

def calculate_export_user_preferences(start: str=Config.train_start, end: str=Config.train_end, n_splits: int=50, save_path: str='./offline_tasks/user_preferences'):
    print('Loading tools...', end='\t')
    dev_raw = [open('./preprocessed/train', 'r').readlines()]
    dev_user_list = []
    for daily in dev_raw:
        users = [u.split()[0] for u in daily]
        dev_user_list.extend(users)

    user_time_read = load_user_time_read(root_dir='./preprocessed/user_time_read.json')
    encoder = PostIdEncoder(root_dir='./encodings')
    tfidf = TFIDFGenerator('./tfidf')
    print('loaded!')

    batch_size = len(dev_user_list) // n_splits

    for i in range(n_splits):
        print(f'Getting User Preferences of Batch #{i}...')
        if i == n_splits-1:
            dev_user_batch = dev_user_list[i*batch_size:]
        else:
            dev_user_batch = dev_user_list[i*batch_size:(i+1)*batch_size]

        posts = pd.DataFrame()
        user_preferences = np.zeros((7000, 1))

        for user_id in tqdm(dev_user_batch, desc=f'Extracting user prefereces based on {start}-{end}'):
            history = filter_read_by_time(user_time_read[user_id], start, end)
            history = squeeze(list(map(lambda x: x[-1], history)))
            
            user_tfidf = tfidf.generate(encoder.transform(history), drop_id=False)
            preference = user_tfidf.drop('post_meta_id', axis=1).values.sum(axis=0)

            posts = pd.concat([posts, user_tfidf], axis=0, ignore_index=True)
            user_preferences = np.hstack([user_preferences, preference.reshape(7000, 1)])

        print('Postprocessing...')
        user_preferences = user_preferences[:, 1:]
        posts = posts.drop_duplicates(ignore_index=True)
        post_meta_id = posts['post_meta_id'].tolist()
        posts = posts.drop('post_meta_id', axis=1)
        idf = np.log(tfidf.DF.values.squeeze()) - np.log((posts != 0).sum().values + 1e-4)

        print('Saving...')
        batch_name = f'({start}-{end})batch{i+1:0>2d}'
        os.mkdir(os.path.join(save_path, batch_name))
        save_npz(os.path.join(save_path, batch_name, f'posts{i+1:0>2d}.npz'), csr_matrix(posts.values))
        save_as_pickle(post_meta_id, os.path.join(save_path, batch_name, f'post_meta_id{i:0>2d}.pkl'))
        np.save(os.path.join(save_path, batch_name, f'idf{i+1:0>2d}.npy'), idf)
        np.save(os.path.join(save_path, batch_name, f'user_preferences{i+1:0>2d}.npy'), user_preferences)


if __name__ == '__main__':
    fire.Fire({'run': calculate_export_user_preferences})