import os
import torch
import pickle
import numpy as np


def encode16(params, save_path):
    """将params压缩到16bit，并保存到save_path"""
    custom_dict = {}
    for name, param in params.items():
        param = np.float64(param.numpy())
        # 有些变量不是ndarray而只是一个数字，这种变量不用压缩
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(save_path, 'wb'))

def decode16(fname):
    '''读取16bit的权重，还原到torch.tensor后以state_dict形式存储'''
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param
    return custom_dict

def encode8(params, save_path):
    """将params压缩到8bit，并保存到save_path"""
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(save_path, 'wb'))

def decode8(fname):
    '''读取8bit的权重，还原到torch.tensor后以state_dict形式存储'''
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param
    return custom_dict


if __name__ == '__main__':
    print(f"Original Cost: {os.stat('./weights/student_model.bin').st_size} Bytes.")
    old_params = torch.load('./weights/student_model.bin', map_location='cpu')
    encode16(old_params, './weights/student_model_16bit.bin')
    print(f"16-bit Cost: {os.stat('./weights/student_model_16bit.bin').st_size} Bytes.")
    encode8(old_params, './weights/student_model_8bit.bin')
    print(f"8-bit Cost: {os.stat('./weights/student_model_8bit.bin').st_size} Bytes.")
