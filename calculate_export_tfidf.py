import os
import pandas as pd

from tf_idf import get_tfidf
from load_data import load
import config


if __name__ == '__main__':
    ROOT = config.data_root
    print('Loading data ...', end='    ')
    metadata = load(name='metadata', root_dir=ROOT)
    data_size = metadata.shape[0]
    vocab = pd.read_csv(os.path.join(ROOT, 'preprocessed/tag_vocab7000.csv'))
    vocab = vocab['tag'].tolist()
    print('loaded!')

    print('Getting TF-IDF...')
    tfidf = get_tfidf(data=metadata, vocab=vocab)
    post_id_list = metadata['post_id'].tolist()
    tfidf.index = post_id_list
    tfidf = tfidf.reset_index().rename({'index': 'post_id'}, axis=1)
    print('Finished!')

    filename = 'metadata_tfidf.csv'
    save_path = os.path.join(ROOT, 'preprocessed', filename)
    tfidf.to_csv(save_path, encoding='euc-kr', index=False)
    print(f'saved as "{save_path}"😎')
    

