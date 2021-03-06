import os
from numpy.core.numeric import indices
import pandas as pd

from tf_idf import get_tfidf
from load_data import load
import config


if __name__ == '__main__':
    SPLIT = True
    ROOT = config.data_root
    
    metadata = load(name='metadata', root_dir=ROOT)
    vocab = pd.read_csv(os.path.join(ROOT, 'preprocessed/tag_vocab7000.csv'))
    vocab = vocab['tag'].tolist()

    if SPLIT:
        n_split = 50
        total_size = metadata.shape[0]
        batch_size = total_size // n_split

        for k in range(n_split+1):
            if k == n_split:
                former, latter = k*batch_size, total_size
            else:
                former, latter = k*batch_size, (k+1)*batch_size
            index_list = [j for j in range(former, latter)]
            save_path = os.path.join(ROOT, 'preprocessed/metadata_tfidf', f'metadata_tfidf_vocab7000_({former}-{latter-1}).npz')
            get_tfidf(data=metadata, vocab=vocab, indices=index_list, save_path=save_path)

    else:
        save_path = os.path.join(ROOT, 'preprocessed/metadata_tfidf', f'metadata_tfidf_vocab7000.npz')
        get_tfidf(data=metadata, vocab=vocab, save_path=save_path)
    

