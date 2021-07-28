import json
import numpy as np
import paddle.fluid as fluid

def iter():
    batch_size = 3
    with open("data/input.json") as f:
        batch_title = []
        batch_click = []
        for line in f:
            info = json.loads(line.strip('\n'))
            title = [int(x) for x in info["item_info"][0]["title"]]
            click = [int(info["item_info"][0]["click"])]
            batch_title.append(title)
            batch_click.append(click)
            if len(batch_title) == batch_size:
                title_data = np.asarray(batch_title, dtype="int64")
                # print("title_data: ", title_data)
                title_data.reshape([batch_size, 1, 1, len(title)])
                '''
                shapes = [len(c) for c in title_data]
                place = fluid.CPUPlace()
                title_data = fluid.create_lod_tensor(title_data.reshape(-1, 1), shapes, place)
                '''
                click_data = np.asarray(batch_click, dtype="int64")
                yield (title_data, click_data)
                batch_title = []
                batch_click = []
