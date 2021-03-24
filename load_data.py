import os
import json
import pickle
from glob import glob

import pandas as pd

from config import Config


def load_user_time_read(root_dir: str) -> dict:
    with open(root_dir, "r") as json_file:
        output = json.load(json_file)
    return output


def load_raw(name: str = "magazine", root_dir: str = Config.raw_dir):
    PATH = {
        "magazine": os.path.join(root_dir, "magazine.json"),
        "metadata": os.path.join(root_dir, "metadata.json"),
        "users": os.path.join(root_dir, "users.json"),
        "dev": os.path.join(root_dir, "predict/dev.users"),
        "test": os.path.join(root_dir, "predict/test.users"),
        "read": os.path.join(root_dir, "read/*"),
    }

    print(f"Loading {name} data ...", end="    ")
    if name in ["magazine", "metadata", "users"]:
        data = pd.DataFrame(
            [json.loads(line) for line in open(PATH[name], "r", encoding="utf-8")]
        )
        if name == "magazine":
            data = data.rename({"id": "magazine_id"}, axis=1)
        elif name == "metadata":
            data.rename({"id": "post_id"}, axis=1, inplace=True)
        else:
            data = data.rename({"id": "user_private"}, axis=1)[
                ["user_private", "following_list", "keyword_list"]
            ]

    elif name in ["dev", "test"]:
        data = pd.read_csv(PATH[name], header=None, names=["user_private"])

    # NOTE: 매우 느림. 업데이트 필요
    elif name == "read":
        try:
            data = pd.concat(
                [_get_read(path) for path in glob(PATH[name])],
                axis=0,
                ignore_index=True,
            )
        except ValueError:
            raise ValueError("Wrong directory.")

    else:
        raise NotImplementedError()
    print("loaded!")

    return data


def _get_read(path: str) -> pd.DataFrame:
    read = pd.read_csv(path, header=None, names=["log"])
    start_time = int(os.path.basename(path).split("_")[0])

    read["user_private"] = read["log"].apply(lambda x: x.split()).apply(lambda x: x[0])
    read["sequence"] = read["log"].apply(lambda x: x.split()).apply(lambda x: x[1:])
    read["start_time"] = start_time

    return read[["start_time", "user_private", "sequence"]]


# class PostIdEncoder:
#     def __init__(self, root_dir: str) -> dict:
#         self.__encoder = self.__load_encoder(root_dir)
#         self.__decoder = self.__load_decoder(root_dir)

#     def transform(self, post_ids: list) -> list:
#         if isinstance(post_ids, str):
#             post_ids = [post_ids]
#         output = [self.__encoder[p] for p in post_ids]

#         if len(output) == 1:
#             return output[0]

#         return output
    
#     def inverse_transform(self, post_meta_ids: list) -> list:
#         if isinstance(post_meta_ids, str):
#             post_meta_ids = [post_meta_ids]

#         output = [self.__decoder[p] for p in post_meta_ids]
#         if len(output) == 1:
#             return output[0]
#         return output

#     def __load_decoder(self, root_dir: str):
#         decoder_path = os.path.join(root_dir, 'post_id_decoding.pickle')
#         with open(decoder_path, "rb") as handle:
#             decoder = pickle.load(handle)
#         return decoder
    
#     def __load_encoder(root_dir: str):
#         encoder_path = os.path.join(root_dir, 'post_id_encoding.pickle')
#         with open(encoder_path, "rb") as handle:
#             encoder = pickle.load(handle)
#         return encoder
        
# def load_post_id_encoder(encoder_dir: str) -> dict:
#     with open(encoder_dir, "rb") as handle:
#         post_id_encoder = pickle.load(handle)
#     return post_id_encoder

# def load_post_id_decoder(decoder_dir: str) -> dict:
#     with open(decoder_dir, "rb") as handle:
#         post_id_decoder = pickle.load(handle)
#     return post_id_decoder

