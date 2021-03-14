from tqdm import tqdm
from collections import defaultdict

import fire

from config import DataRoots
from load_data import load_raw, load_post_id_encoder
from utils import save_as_json



def reconstruct_reads_by_user(
    read_path: str = DataRoots.raw,
    post_enc_path: str = DataRoots.post_id_enc,
    save_path: str = None,
) -> dict:
    encoder = load_post_id_encoder(encoder_dir=post_enc_path)
    reads = load_raw(name="read", root_dir=read_path)
    DATE, SEQUENCE = 0, 1

    user_list = reads["user_private"].unique().tolist()
    
    users_log_dict = defaultdict(list)
    for user_id in tqdm(user_list):
        logs = reads.loc[reads["user_private"] == user_id].drop(
            "user_private", axis=1
        )
        logs = logs.apply(lambda x: {x[DATE]: _encode_post_id_sequence(x[SEQUENCE], encoder)}, axis=1).tolist()
        users_log_dict[user_id].append(logs)
            
    if save_path:
        save_as_json(users_log_dict, save_path)
    else:
        return users_log_dict

def _encode_post_id_sequence(sequence: str, encoder: str):
    post_id_sequence = [encoder[x] for x in sequence if x in encoder.keys()]
    return post_id_sequence


if __name__ == '__main__':
    fire.Fire({'preprocess_read': reconstruct_reads_by_user})