import ast


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
    '''2차원 리스트를 1차원으로 squeeze'''
    result = []
    for l in arr:
        if len(l) > 0 and isinstance(l, list):
            result.extend(l)
        elif not isinstance(l, list):
            result.append(l)
    return result