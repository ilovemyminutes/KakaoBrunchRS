import os
from numpy.core.numeric import indices
import pandas as pd

from tf_idf import get_tfidf
from load_data import load
import config


if __name__ == '__main__':
    ROOT = config.data_root
    
    metadata = load(name='metadata', root_dir=ROOT)
    n_split = 100
    total_size = metadata.shape[0]
    batch_size = total_size // n_split
    
    vocab = pd.read_csv(os.path.join(ROOT, 'preprocessed/tag_vocab7000.csv'))
    vocab = vocab['tag'].tolist()

    for k in range(n_split+1):
        if k == n_split:
            former, latter = k*batch_size, total_size
        else:
            former, latter = k*batch_size, (k+1)*batch_size
        index_list = [j for j in range(former, latter)]
        save_path = os.path.join(ROOT, 'preprocessed/metadata_tfidf', f'metadata_tfidf_({former}-{latter-1}).csv')
        get_tfidf(data=metadata, vocab=vocab, indices=index_list, save_path=save_path)
    

