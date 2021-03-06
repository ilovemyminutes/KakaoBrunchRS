import os
import pandas as pd

import fire

from tfidf import get_tfidf
from load_data import load
import config

ROOT = config.data_root
VOCAB_PATH = os.path.join(ROOT, "preprocessed/tag_vocab7000.csv")


def get_save_tfidf(
    root_dir: str = ROOT,
    vocab_path: str = VOCAB_PATH,
    n_splits: int = 50,
    save_dir="test",
):
    metadata = load(name="metadata", root_dir=root_dir)
    vocab = pd.read_csv(vocab_path)["tag"].tolist()

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

    else:
        filename = "metadata_tfidf_vocab7000.npz"
        save_path = os.path.join(save_dir, filename)
        get_tfidf(data=metadata, vocab=vocab, save_path=save_path)


if __name__ == "__main__":
    fire.Fire({"run": get_save_tfidf})
