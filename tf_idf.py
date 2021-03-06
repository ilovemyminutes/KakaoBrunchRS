import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import squeeze


def get_tfidf(data: pd.DataFrame, vocab: list, indices: list = None, save_path: str=None, encoding: str='euc-kr') -> pd.DataFrame:
    """tf-idf matrix를 리턴하는 함수. 서브샘플링을 통해 일부 데이터셋에 대해서만 tf-idf를 구할 수 있음
    tf-idf 방식
        - tf: boolean
        - idf: logarithmic
        - Reference: https://ko.wikipedia.org/wiki/Tf-idf
    Args:
        data (pd.DataFrame): 전체 데이터셋
        vocab (list): TF-IDF를 매길 태그 리스트
        indices (list, optional): 샘플링할 경우 활용. iloc에 사용될 인덱스 리스트를 입력(정수로 단일 샘플링도 가능). Defaults to None.

    Returns:
        pd.DataFrame: 사이즈가 (data size, vocab_size)인 TF-IDF matrix
    """
    if isinstance(indices, int):
        indices = [indices]

    num_docs = data.shape[0]
    batch = copy.deepcopy(data) if indices is None else data.iloc[indices, :]

    idf = get_idf(batch=batch, vocab=vocab, num_docs=num_docs)
    tf = get_tf(batch=batch, vocab=vocab)

    print("Aggregating TF and IDF...")
    output = tf * idf.values # broad-casting

    if save_path is not None:
        print('Saving TF-IDF...', end='    ')
        post_id_list = batch['post_id'].tolist()
        output.index = post_id_list
        output = output.reset_index().rename({'index': 'post_id'}, axis=1)
        output.to_csv(save_path, encoding=encoding, index=False)
        print(f'saved as "{save_path}"😎')
    else:
        return output


def get_idf(batch: pd.DataFrame, vocab: list, num_docs: int) -> pd.DataFrame:
    """batch 데이터의 IDF 리턴
        - IDF 방식: logarithmic
        - Reference: https://ko.wikipedia.org/wiki/Tf-idf

    Args:
        batch (pd.DataFrame): metadata의 일부
        vocab (list): IDF에 활용할 키워드 리스트
        num_docs (int): metadata(모집단)의 모든 문서, 즉 행의 개수

    Returns:
        pd.DataFrame: (1, vocab 수) 크기의 IDF 행렬
    """    
    vocab_size = len(vocab)
    idf = pd.DataFrame(np.zeros((1, vocab_size)), columns=vocab)

    kwd_list = set(squeeze(batch["keyword_list"].tolist()))
    kwd_list_filtered = list(filter(lambda x: x in vocab, kwd_list))

    pbar = tqdm(kwd_list_filtered)
    pbar.set_description("Getting IDF")
    for tag in pbar:
        if tag in vocab:
            idf[tag] = np.log(num_docs) - np.log(
                sum(batch["keyword_list"].apply(lambda x: tag in x))
            )

    return idf


def get_tf(batch: pd.DataFrame, vocab: list) -> pd.DataFrame:
    """batch 데이터의 tf를 리턴
        - tf 방식: boolean
        - Reference: https://ko.wikipedia.org/wiki/Tf-idf

    Args:
        batch (pd.DataFrame): metadata의 일부
        vocab (list): TF에 활용할 키워드 리스트

    Returns:
        pd.DataFrame: (batch 데이터 row 수, vocab 수) 크기의 TF 행렬
    """    
    vocab_size = len(vocab)
    tf = pd.DataFrame(np.zeros((batch.shape[0], vocab_size)), columns=vocab)

    # split two cases because of vesion issue
    try:
        tqdm.pandas(desc="Getting TF")
        tf = tf.progress_apply(lambda x: _sparkle(x, batch, vocab), axis=1)
    except ImportError:
        print("Getting TF...")
        tf = tf.apply(lambda x: _sparkle(x, batch, vocab), axis=1)

    return tf


def _sparkle(row: pd.DataFrame, batch: pd.DataFrame, vocab: list):
    """boolean-type TF를 구하는 함수. get_tfidf() 내부에서 다음과 같은 형태로 활용

        output.apply(lambda x: onehot(x, sample), axis=1)

    Args:
        sample_row (pd.Series): apply 함수가 적용된 샘플 데이터의 각 행
        sample (pd.DataFrame): 샘플 데이터. 키워드 리스트 정보를 얻기 위해 활용

    Returns:
        [type]: [description]
    """
    idx = row.name
    for tag in batch["keyword_list"].iloc[idx]:
        if tag in vocab:
            row[tag] += 1
    return row


# deprecated: too slow
# def get_tfidf(data, vocab: list, indices: list=None, tf_type: str='boolean'):
#     if isinstance(indices, int):
#         indices = [indices]
#     vocab_size = len(vocab)
#     num_docs = data.shape[0]
#     sample = copy.deepcopy(data) if indices is None else data.iloc[indices, :]
#     output = pd.DataFrame(np.zeros((sample.shape[0], vocab_size)), columns=vocab)

#     if tf_type == 'boolean':
#         tf = 1 # boolean tf
#         output.apply(lambda )
#         for _, row in sample.iterrows():
#             tfidf = dict().fromkeys(vocab)
#             for tag in row['keyword_list']:
#                 idf = np.log(num_docs / sum(sample['keyword_list'].apply(lambda x: tag in x)))
#                 tfidf[tag] = tf * idf
#             output = output.append(tfidf, ignore_index=True)
#     else:
#         raise NotImplementedError()

#     return output
