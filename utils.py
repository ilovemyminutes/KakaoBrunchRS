import os
import ast
from glob import glob

import pandas as pd
from scipy.sparse import load_npz, vstack


def load_tfidf(tfidf_dir: str, vocab_dir: str, post_meta_id_list: list) -> pd.DataFrame:
    if isinstance(post_meta_id_list, int):
        post_meta_id_list = [post_meta_id_list]

    tfidf = vstack([load_npz(split) for split in glob(os.path.join(tfidf_dir, '*'))])
    vocab = pd.read_csv(vocab_dir)['tag'].tolist()
    columns = ['post_meta_id'] + vocab
    
    output = pd.DataFrame(tfidf[post_meta_id_list, :].todense(), columns=columns)
    output['post_meta_id'] = output['post_meta_id'].astype(int)

    return output



def str2list(strlist: str) -> list:
    """문자열 형태의 리스트를 리스트로 인식되도록 하는 함수
    예: "['a', 'b', 'c', 'd']" -> ['a', 'b', 'c', 'd']

    Args:
        strlist (str): 문자열 형태의 리스트

    Returns:
        list: 리스트 타입으로 인식된 리스트
    """
    listed = ast.literal_eval(strlist)
    return listed


def squeeze(arr: list) -> list:
    """2차원 리스트를 1차원으로 squeeze

    Args:
        arr (list): 2차원 리스트. 각 원소는 int, float, list 중 하나의 자료형을 가짐

    Returns:
        list: 1차원 리스트. 각 원소는 int 또는 float
    """
    result = []
    for l in arr:
        if len(l) > 0 and isinstance(l, list):
            result.extend(l)
        elif not isinstance(l, list):
            result.append(l)
    return result
