import json
import numpy as np
import paddle.fluid as fluid

def iter():
    batch_size = 1
    with open("data/input.json") as f:
        batch_uid = []
        batch_title = []
        batch_click = []
        batch_action = []
        for line in f:
            info = json.loads(line.strip('\n'))
            uid = info["user_info"]["uid"]
            title = [int(x) for x in info["item_info"][0]["title"]]
            if len(title) < 12:
                title += [0 for _ in range(12 - len(title))]
            action = [int(x) for x in info["item_info"][0]["action"]]
            if len(action) < 12:
                action += [0 for _ in range(12 - len(action))]
            click = [int(info["item_info"][0]["click"])]
            batch_uid.append(uid)
            batch_title.append(title)
            batch_click.append(click)
            batch_action.append(action)
            if len(batch_title) == batch_size:
                title_data = np.asarray(batch_title, dtype="int64")
                title_data.reshape([batch_size, 1, 1, len(title)])
                action_data = np.asarray(batch_action, dtype="int64")
                action_data.reshape([batch_size, 1, 1, len(action)])
                click_data = np.asarray(batch_click, dtype="int64")
                yield (batch_uid, title_data, action_data, click_data)
                batch_uid = []
                batch_title = []
                batch_click = []
                batch_action = []
