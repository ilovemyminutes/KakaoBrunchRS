from tqdm import tqdm
from collections import defaultdict, ChainMap

import pandas as pd
import fire

from config import DataRoots
from load_data import load_raw, load_post_id_encoder
from utils import save_as_json


class UserLogsGenerator:

    DATE, SEQUENCE = 0, 1

    def __init__(
        self, read_path: str = DataRoots.raw, post_enc_path: str = DataRoots.post_id_enc, period: tuple= None
    ):  
        if period is not None:
            raise NotImplementedError()

        self.read_path = read_path
        self.post_enc_path = post_enc_path

        self.__reads = load_raw(name="read", root_dir=read_path)
        self.__encoder = load_post_id_encoder(post_enc_path)
        self.__user_list = self.__reads["user_private"].unique().tolist()

        
    def get_user_log(
        self, user_list: list, save_path: str = None
    ) -> pd.DataFrame:
        
        user_logs = defaultdict(dict)
        for user in tqdm(user_list):
            logs_raw = self.__reads.loc[self.__reads["user_private"] == user].drop(
                "user_private", axis=1
            )
            logs_raw = logs_raw.apply(
                lambda x: {
                    x[self.DATE]: self._encode_post_id_sequence(x[self.SEQUENCE], self.__encoder)
                },
                axis=1,
            ).tolist()
            logs = dict(ChainMap(*logs_raw))
        user_logs[user] = logs

        if save_path:
            save_as_json(user_logs, save_path)
            return user_logs
        else:
            return user_logs

    @staticmethod
    def _encode_post_id_sequence(sequence: list, encoder: dict):
        post_id_sequence = [encoder[x] for x in sequence if x in encoder.keys()]
        return post_id_sequence

    @property
    def read(self):
        return self.__reads

    @property
    def encoder(self):
        return self.__encoder

    @property
    def user_list(self):
        return self.__user_list


if __name__ == "__main__":
    fire.Fire({"generate": UserLogsGenerator})
