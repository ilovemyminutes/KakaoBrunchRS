from load_data import load_raw, load_post_id_decoder, load_post_id_decoder
from config import DataRoots


def remake_reads_by_user(reads_root: str, post_enc_root: str, save: bool=True) -> dict:
    reads = load_raw(name='read')
    users_log_dict = defaultdict(list)

    for user_id in tqdm(user_private_list):
        user_reads = reads_subset.loc[reads_subset['user_private']==user_id]
        user_reads = user_reads.drop('user_private', axis=1)

        for _, row in user_reads.iterrows():
            time_piece = {row['start_time']: encode_post_id_sequence(row['sequence'])}
            users_log_dict[user_id].append(time_piece)


def _encode_post_id_sequence(sequence):
    post_id_list = list(post_id_encoding.keys())
    post_id_sequence = [post_id_encoding[x] for x in sequence if x in post_id_list]
    return post_id_sequence

def get_post_id_encoder(post_enc_root: str=DataRoots.post_id_enc):
    return