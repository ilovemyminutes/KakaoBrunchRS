from dataclasses import dataclass

# data_root = "./raw"
# tfidf_root = '../raw/preprocessed/metadata_tfidf'
# vocab_root = '../raw/preprocessed/tag_vocab7000.csv'


@dataclass
class DataRoots:
    raw: str = "./raw"
    tfidf: str = (
        "./raw/preprocessed/metadata_tfidf/metadata_tfidf_vocab7000_aggregation.npz"
    )
    vocab: str = "./raw/preprocessed/tag_vocab7000.csv"
    post_id_dec: str = "./raw/preprocessed/post_id_decoding.pickle"
    post_id_enc: str = "./raw/preprocessed/post_id_encoding.pickle"
