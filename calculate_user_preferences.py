import os
from tqdm import tqdm
import numpy as np
from scipy import sparse
import fire
from load_data import load_user_time_read
from preprocessing import PostIdEncoder
from tfidf import TFIDFGenerator
from utils import squeeze, load_pickle
from config import Config


def calculate_user_preferences(user_id_list: list=None, start: str=Config.train_start, end:str=Config.train_end, save_path: str='./recommendation_outputs/'):
    if user_id_list == 'dev':
        user_id_list = load_pickle(Config.dev_user_list)
    elif user_id_list == 'test':
        user_id_list = load_pickle(Config.test_user_list)

    print('Load calculating tools...', end='\t')
    user_time_read = load_user_time_read(Config.user_time_read)
    post_id_encoder = PostIdEncoder(Config.encodings_root)
    tfidf_generator = TFIDFGenerator(Config.tfidf_root)
    print('loaded!')

    post_meta_id = [] # user_id_list의 유저들이 조회한 모든 글
    posts_raw = [] # 유저들의 로그에서 등장한 모든 글을 담을 리스트
    user_preferences_raw = [] # 유저들의 feature 벡터를 담을 리스트

    for user_id in tqdm(user_id_list, desc=f'User Preference Extraction ({start}-{end})'):
        # 설정한 구간에 대한 해당 유저의 로그
        history = _filter_read_by_time(user_time_read[user_id], start, end) 
        history = squeeze(list(map(lambda x: x[-1], history)))

        # 유저 로그로부터 TF-IDF 행렬 생성
        user_tfidf = tfidf_generator.generate(post_id_encoder.transform(history), drop_id=False) # 

        # TF-IDF 행렬로부터 유저 feature 벡터를 생성
        preference = sparse.csr_matrix(user_tfidf.iloc[:, 1:].values.sum(axis=0)[:, np.newaxis]) # post_meta_id 컬럼을 제외한 뒤 summation
        user_tfidf = user_tfidf.groupby('post_meta_id').first().reset_index() # faster than drop_duplicates()
        user_tfidf = user_tfidf.loc[~user_tfidf['post_meta_id'].isin(post_meta_id), :]
        if len(user_tfidf) > 0:
            post_meta_id.extend(user_tfidf['post_meta_id'].tolist())
        posts_raw.append(sparse.csr_matrix(user_tfidf.iloc[:, 1:])) # post_meta_id 컬럼을 제외하고 append -> post_meta_id 리스트를 개별적으로 생성하므로 불필요
        user_preferences_raw.append(preference)

    print('Postpreprocessing...', end='\t')
    posts = sparse.vstack(posts_raw)
    user_preferences = sparse.hstack(user_preferences_raw)
    idf = np.array(np.log(tfidf_generator.DF.values.squeeze()) - np.log((posts != 0).sum(axis=0) + 1e-4)) # 1e-4: to prevent ZeroDivisionError
    recommend_output = (posts.multiply(idf)).dot(user_preferences)
    print('finished!')

    if save_path:
        sparse.save_npz(os.path.join(save_path, 'posts.npz'), posts)
        sparse.save_npz(os.path.join(save_path, 'user_preferences.npz'), user_preferences)
        np.save(os.path.join(save_path, 'idf.npy'), idf)
        sparse.save_npz(os.path.join(save_path, 'recommend_output.npz'), recommend_output)
        print(f'Saved successfully in {save_path}😎')
    else:
        return recommend_output, user_preferences, idf, posts
        
def _filter_read_by_time(history, start, end):
    history_filtered = list(filter(lambda x: (int(start) <= int(x[0].split('_')[0])) and (int(end) >= int(x[0].split('_')[-1])), history))
    return history_filtered


if __name__ == '__main__':
    fire.Fire({'run': calculate_user_preferences})