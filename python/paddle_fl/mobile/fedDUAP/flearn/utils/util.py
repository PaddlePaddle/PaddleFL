"""
util
"""
import os
import re

# 获取当前目录
cwd = os.getcwd()
if cwd.find("/") == -1:
    cwd += "\\"
    root_name = re.search(r"(Fed.*?\\)", cwd).group(1)
else:
    cwd += "/"
    root_name = re.search(r"(Fed.*?/)", cwd).group(1)

# 获取根目录
root_path = cwd[:cwd.find(root_name) + len(root_name)]
result_path = os.path.join(root_path, "result")


def save_result(args, file_path, data):
    """
    保存结果
    :param file_path:
    :param data:
    :return:
    """
    file_path = os.path.join(result_path, args.dataset, args.model, file_path)
    dir_name = os.path.dirname(file_path)
    # print(f"save to {file_path}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_path, "w") as f:
        f.write(data)

def record_log(args, file_path, data):
    """
    记录日志
    :param file_path:
    :param data:
    :return:
    """
    file_path = os.path.join(result_path, args.dataset, args.model, file_path)
    dir_name = os.path.dirname(file_path)
    # print(f"log to {file_path}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_path, "a") as f:
        f.write(data)