import os
import pandas as pd

from tf_idf import get_tfidf
from load_data import load
import config


if __name__ == "__main__":
    ROOT = config.data_root
    metadata = load("metadata")

    vocab = pd.read_csv(os.path.join(ROOT, "preprocessed/tag_vocab7000.csv"))
    tfidf = get_tfidf(data=metadata, vocab=vocab, indices=[1, 2])

    post_id_list = metadata["post_id"].tolist()
    tfidf = tfidf.reset_index().rename({"index": "post_id"}, axis=1)

    save_path = os.path.join(ROOT, "preprocessed/metadata_tfidf.csv")
    tfidf.to_csv(save_path, encoding="euc-kr")
