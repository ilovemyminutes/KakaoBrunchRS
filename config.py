from dataclasses import dataclass
# data_root = "./raw"
# tfidf_root = '../raw/preprocessed/metadata_tfidf'
# vocab_root = '../raw/preprocessed/tag_vocab7000.csv'

@dataclass
class Config:
    data_root: str="./raw"
    tfidf_root: str='../raw/preprocessed/metadata_tfidf/metadata_tfidf_vocab7000_aggregation.npz'
    vocab_root: str='../raw/preprocessed/tag_vocab7000.csv'