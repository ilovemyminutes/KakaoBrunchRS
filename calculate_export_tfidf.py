import os
import pickle
import pandas as pd

import fire

from tfidf import get_tfidf, get_df
from load_data import load_raw
from utils import save_as_pickle
from config import Config


VOCAB_PATH = os.path.join(Config.tfidf_dir, Config.vocab)

def get_save_tfidf(
    root_dir: str = Config.raw_dir,
    vocab_path: str = VOCAB_PATH,
    n_splits: int = 50,
    save_dir="test",
):
    metadata = load_raw(name="metadata", root_dir=root_dir)
    with open(vocab_path, "rb") as tag7000:
        vocab = pickle.load(tag7000)

    # TF-IDF를 분할저장. 코랩 프로 기준 전체 metadata로 TFIDF를 구할 경우 killed 현상이 발생
    if n_splits:
        total_size = metadata.shape[0]
        batch_size = total_size // n_splits

        for k in range(n_splits + 1):
            if k == n_splits:
                former, latter = k * batch_size, total_size
            else:
                former, latter = k * batch_size, (k + 1) * batch_size
            index_list = [j for j in range(former, latter)]

            filename = f"metadata_tfidf_vocab7000_({former}-{latter-1}).npz"
            save_path = os.path.join(save_dir, filename)
            get_tfidf(
                data=metadata, vocab=vocab, indices=index_list, save_path=save_path
            )

    # 전체 metadata로부터 한번에 TFIDF를 생성
    else:
        filename = "metadata_tfidf_vocab7000.npz"
        save_path = os.path.join(save_dir, filename)
        get_tfidf(data=metadata, vocab=vocab, save_path=save_path)


def get_save_df(root_dir: str=Config.raw_dir):
    metadata = load_raw('metadata', root_dir)
    with open(VOCAB_PATH, "rb") as tag7000:
        vocab = pickle.load(tag7000)
    df = get_df(metadata, vocab)
    save_as_pickle(df.values.squeeze().tolist(), '../tfidf/df_vocab7000_aggregation.pkl')


if __name__ == "__main__":
    fire.Fire({"tfidf": get_save_tfidf, 'df': get_save_df})
