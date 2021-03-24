import os
from tqdm import tqdm
import pandas as pd
from scipy import sparse
import fire
from load_data import load_user_time_read
from utils import load_pickle, squeeze
from user_preference import filter_read_by_time
from preprocessing import PostIdEncoder
from config import Config


def recommend(recommend_src: str=Config.recommend_src, output_root: str=Config.output_root):
    print('Loading recommendation tools...', end='\t')
    user_time_read = load_user_time_read(Config.user_time_read)
    user_id_list = load_pickle(Config.test_user_list)
    recommend_meta = sparse.load_npz(os.path.join(recommend_src, 'recommend_output.npz'))
    post_meta_id = load_pickle(os.path.join(recommend_src, 'post_meta_id.pkl'))
    encoder = PostIdEncoder(Config.encodings_root)
    print('loaded!')

    with open(os.path.join(output_root, 'recommend.txt'), 'w') as file:
        for idx in tqdm(range(len(user_id_list)), desc='Generate Recommendations'):
            user_id = user_id_list[idx]
            
            seens = filter_read_by_time(user_time_read[user_id], start=Config.train_start, end=Config.train_end)
            seens = squeeze(list(map(lambda x: x[-1], seens)))
            
            recommend_raw = recommend_meta[:, idx]
            recommend_raw = pd.Series(recommend_raw.toarray().flatten(), index=post_meta_id).sort_values(ascending=False)
            recommend_raw = recommend_raw[~recommend_raw.index.isin(seens)].head(100).index.tolist()
            recommend = list(map(lambda x: encoder.inverse_transform(x), recommend_raw))
            
            file.write(f'{user_id} ')
            file.write(f"{' '.join(recommend)}\n")
            
    print(f"Saved 'recommend.txt' in {output_root}")


if __name__ == '__main__':
    fire.Fire({'run': recommend})