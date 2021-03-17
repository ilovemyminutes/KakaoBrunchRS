import os
from tqdm import tqdm
from collections import defaultdict
from glob import glob

import fire
from config import DataRoots

def get_user_time_read(root_dir: str=DataRoots.raw, save_as: str='./raw/user_time_read.json'):
    paths = sorted(glob(os.path.join(root_dir, "read/*")))
    user_time_read = defaultdict(list)

    for path in tqdm(paths):
        timestamp = os.path.basename(path)
        raw_logs = list(map(lambda x: x[:-1], open(path, 'r').readlines()))
        users = list(map(lambda x: x.split()[0], raw_logs))
        sequences = list(map(lambda x: x.split()[1:], raw_logs))

        for user, seq in zip(users, sequences):
            logs = [timestamp, seq]
            user_time_read[user].append(logs)
    
    if save_as is not None:
        import json
        with open(save_as, "w") as json_file:
            json.dump(user_time_read, json_file)

if __name__ == '__main__':
    fire.Fire({'run':get_user_time_read})