from dataclasses import dataclass

# data_root = "./raw"
# tfidf_root = '../raw/preprocessed/metadata_tfidf'
# vocab_root = '../raw/preprocessed/tag_vocab7000.csv'


@dataclass
class Config:
    raw_dir: str = "./raw"
    user_time_read: str = "./preprocessed/user_time_read.json"
    encodings_root: str = "./encodings/"

    tfidf_root: str = "./tfidf"
    tfidf: str = 'metadata_tfidf_vocab7000_aggregation.npz'
    df: str = 'df_vocab7000_aggregation.pkl'
    vocab: str = 'tag_vocab7000.pkl'

    train_start: str= '2018100100'
    train_end: str= '2019022200'
    dev_start: str= '2019022200'
    dev_end: str= '2019030100'