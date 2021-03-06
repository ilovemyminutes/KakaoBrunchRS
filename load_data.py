import os
import json
from glob import glob
import pandas as pd


def get_read(path: str) -> pd.DataFrame:
    read = pd.read_csv(path, header=None, names=["log"])
    start_time = int(os.path.basename(path).split("_")[0])

    read["user_private"] = read["log"].apply(lambda x: x.split()).apply(lambda x: x[0])
    read["sequence"] = read["log"].apply(lambda x: x.split()).apply(lambda x: x[1:])
    read["start_time"] = start_time

    return read[["start_time", "user_private", "sequence"]]


def load(name: str = "magazine", root_dir: str = "../raw/"):
    PATH = {
        "magazine": os.path.join(root_dir, "magazine.json"),
        "metadata": os.path.join(root_dir, "metadata.json"),
        "users": os.path.join(root_dir, "users.json"),
        "dev": os.path.join(root_dir, "predict/dev.users"),
        "test": os.path.join(root_dir, "predict/test.users"),
        "read": os.path.join(root_dir, "read/*"),
    }

    print(f'Loading {name} data ...', end='    ')
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

    elif name == "read":
        data = pd.concat(
            [get_read(path) for path in glob(PATH[name])], axis=0, ignore_index=True
        )

    else:
        raise NotImplementedError()
    print('loaded!')

    return data
