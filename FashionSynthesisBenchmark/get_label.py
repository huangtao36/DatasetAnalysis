import h5py
from scipy.io import loadmat
import cv2
import numpy as np
import os
import torch
import shutil
from PIL import Image
import pickle
from nltk.tokenize import RegexpTokenizer  # use to split sentence to words


def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


def flatten(arr):
    if isinstance(arr, list):
        return [a for i in arr for a in flatten(i)]
    else:
        return [arr]


def get_index(list_, value):
    """
    双层列表中获得value所在第一层的index
    :param list_:
    :param value:
    :return:
    """
    for idx in range(len(list_)):
        if isinstance(list_[idx], list):
            for a_idx in range(len(list_[idx])):
                if value == list_[idx][a_idx]:
                    return idx
        else:
            if value  == list_[idx]:
                return idx
    return None


# 性别
Male = ['man', 'boy']
Female = ['lady', 'woman', 'girl']
GENEDER = [Male, Female]  # 2类


# 款式
STYLE = ['tee', 'blouse', 'tank', 'sweater',
       'top',  'hoodie', 'henley', 'jersey', 'jacket',
         'blazer', 'bomber', 'parka', 'coat', 'cardigan',
         'dress', 'romper', 'jumpsuit', 'kimono', 'poncho']

# 颜色
Multicolor = ['multicolor', 'multi', 'multicolored']
COLOR = ['white', 'silver', 'gray', 'black',
         'red', 'blue', 'green', 'olive', 'beige',
         'yellow', 'khaki', 'orange','purple',
         'pink', 'brown',
         Multicolor]

# 袖子
SLEEVE = ['long', 'sleeveless', 'short']  # 3类

# dataroot = '/home/OpenResource/Datasets/DeepFashion/FashionSynthesisBenchmark'
dataroot = r'F:\Datas\DeepFashion\Fashion Synthesis Benchmark'
anno = loadmat(os.path.join(dataroot, 'Anno/language_original.mat'))

index_data = loadmat(os.path.join(dataroot, 'Eval/ind.mat'))
# img_and_parser = h5py.File(os.path.join(dataroot, 'Img/G2.h5'), 'r')
# key_points_file = os.path.join(dataroot, 'peaks.pickle')
# with open(key_points_file, mode='rb') as f:
#     peaks_dic = pickle.load(f)


# 给每一个图像添加标签
# LABEL = {'geneder': -1, 'style': -1, 'color': -1, 'sleeve': -1}

length = anno['engJ'].shape[0]

save_dic = {}
for idx in range(length):
    words = split_sentence_into_words(anno['engJ'][idx][0][0])
    # print(words)
    img_file = anno['nameList'][idx][0][0]
    LABEL = {'geneder': -1, 'style': -1, 'color': -1, 'sleeve': -1}
    for num in range(len(words)):
        word = words[num]
        # geneder
        geneder_idx = get_index(GENEDER, word)
        if geneder_idx is not None:
            LABEL['geneder'] = geneder_idx

        # style
        style_idx = get_index(STYLE, word)
        if style_idx is not None:
            LABEL['style'] = style_idx

        # color
        color_idx = get_index(COLOR, word)
        if color_idx is not None:
            LABEL['color'] = color_idx

        # sleeve
        sleeve_idx = get_index(SLEEVE, word)
        if sleeve_idx is not None:
            LABEL['sleeve'] = sleeve_idx

    flag = 0
    for value in LABEL.values():
        if value == -1:
            flag = 1
        else: pass
    # print(LABEL)

    if flag == 1:
        continue

    # print(LABEL)
    save_dic[img_file] = LABEL

    if idx % 1000 == 0:
        print('{}'.format(idx))
    #     break

with open('./class_label.pkl', mode='wb') as f:
    pickle.dump(save_dic, f)
