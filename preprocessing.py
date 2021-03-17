import os
import pickle
from tqdm import tqdm
from collections import defaultdict, ChainMap

import pandas as pd
import fire

from config import DataRoots
from load_data import load_raw
from utils import save_as_json


class PostIdEncoder:
    def __init__(self, root_dir: str) -> dict:
        self.__encoder = self.__load_encoder(root_dir)
        self.__decoder = self.__load_decoder(root_dir)

    def transform(self, post_ids: list) -> list:
        if isinstance(post_ids, str):
            post_ids = [post_ids]

        output = [self.__encoder[p] for p in post_ids if p in self.__encoder.keys()]
        return output
    
    def inverse_transform(self, post_meta_ids: list) -> list:
        if isinstance(post_meta_ids, str):
            post_meta_ids = [post_meta_ids]

        output = [self.__decoder[p] for p in post_meta_ids if p in self.__decoder.keys()]
        return output

    def __load_decoder(self, root_dir: str):
        decoder_path = os.path.join(root_dir, 'post_id_decoding.pickle')
        with open(decoder_path, "rb") as handle:
            decoder = pickle.load(handle)
        return decoder
    
    def __load_encoder(self, root_dir: str):
        encoder_path = os.path.join(root_dir, 'post_id_encoding.pickle')
        with open(encoder_path, "rb") as handle:
            encoder = pickle.load(handle)
        return encoder


# class UserLogsGenerator:
#     DATE, SEQUENCE = 0, 1

#     def __init__(
#         self, read_path: str = DataRoots.raw, post_enc_path: str = DataRoots.post_id_enc, period: tuple= None
#     ):  
#         if period is not None:
#             raise NotImplementedError()

#         self.read_path = read_path
#         self.post_enc_path = post_enc_path

#         self.__reads = load_raw(name="read", root_dir=read_path)
#         self.__encoder = load_post_id_encoder(post_enc_path)
#         self.__user_list = self.__reads["user_private"].unique().tolist()

        
#     def get_user_log(
#         self, user_list: list, save_path: str = None
#     ) -> pd.DataFrame:
        
#         user_logs = defaultdict(dict)
#         for user in tqdm(user_list):
#             logs_raw = self.__reads.loc[self.__reads["user_private"] == user].drop(
#                 "user_private", axis=1
#             )
#             logs_raw = logs_raw.apply(
#                 lambda x: {
#                     x[self.DATE]: self._encode_post_id_sequence(x[self.SEQUENCE], self.__encoder)
#                 },
#                 axis=1,
#             ).tolist()
#             logs = dict(ChainMap(*logs_raw))
#         user_logs[user] = logs

#         if save_path:
#             save_as_json(user_logs, save_path)
#             return user_logs
#         else:
#             return user_logs

#     @staticmethod
#     def _encode_post_id_sequence(sequence: list, encoder: dict):
#         post_id_sequence = [encoder[x] for x in sequence if x in encoder.keys()]
#         return post_id_sequence

#     @property
#     def read(self):
#         return self.__reads

#     @property
#     def encoder(self):
#         return self.__encoder

#     @property
#     def user_list(self):
#         return self.__user_list


if __name__ == "__main__":
    fire.Fire({"generate": UserLogsGenerator})
