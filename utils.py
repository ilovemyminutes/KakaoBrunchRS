import os
import ast
import pickle
from config import Config

def load_post_id_encoder(encoder_dir: str) -> dict:
    with open(encoder_dir, 'rb') as handle:
        post_id_encoder = pickle.load(handle)
    return post_id_encoder

def load_post_id_decoder(decoder_dir: str) -> dict:
    with open(decoder_dir, 'rb') as handle:
        post_id_decoder = pickle.load(handle)
    return post_id_decoder

def save_pickle(f: object, save_path: str) -> None:
    with open(save_path, 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{f} saved in {save_path} successfuly.')


def iterate_data_files(from_dtm, to_dtm, root_dir: str=Config.data_root):
    LENGTH = len('YYYYMMDDHH_YYYYMMDDHH')
    from_dtm, to_dtm = map(str, [from_dtm, to_dtm])
    read_root = os.path.join(root_dir, "read")
    for fname in os.listdir(read_root):
        if len(fname) != LENGTH:
            continue
        if from_dtm != "None" and from_dtm > fname:
            continue
        if to_dtm != "None" and fname > to_dtm:
            continue
        path = os.path.join(read_root, fname)
        yield path, fname


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
