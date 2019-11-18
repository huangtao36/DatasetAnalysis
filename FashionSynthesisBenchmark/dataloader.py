from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from scipy.io import loadmat
from collections import Counter
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as DATA
from PIL import Image, ImageOps

import os
import cv2 as cv
import math
import sys
import h5py
import numpy as np
from PIL import Image
import numpy.random as random
import pickle
from nltk.tokenize import RegexpTokenizer


def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())

def get_caption(sentence, wordtoix, word_num=25, word_embedding=None):
    words = split_sentence_into_words(sentence)

    if word_embedding is not None:
        word_vecs = torch.Tensor([word_embedding[w] for w in words])  # vector [lens, 300]
        # zero padding to 25
        if len(words) < word_num:  # 25
            word_vecs = torch.cat((
                word_vecs,
                torch.zeros(word_num - len(words), word_vecs.size(1))
            ))
        len_desc = len(words)
        return word_vecs, len_desc
    else:
        sent_caption = np.asarray(
            [wordtoix[w] for w in words]).astype('int64')

        # a list of indices for a sentence
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((word_num, 1), dtype='int64')
        x_len = num_words
        if num_words <= word_num:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:word_num]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = word_num

        return x, x_len



def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            distance = np.sqrt(float(i ** 2 + j ** 2))
            if (r + i) >= 0 and (r + i) < height and (c + j) >= 0 and (
                    c + j) < width:
                if 'Solid' == mode and distance <= radius:
                    indices.append([r + i, c + j, k])

    return indices


def _getSparsePose(peaks, basic_h, basic_w, height, width,
                   channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []

    h_mark = height / basic_h
    w_mark = width / basic_w
    radius = radius * int(h_mark)

    for k in range(len(peaks)):
        p = peaks[k]
        if 0 != len(p):
            r = p[0][1]
            r = r * h_mark
            c = p[0][0]
            c = c * w_mark
            ind = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)

    shape = [height, width, channel]
    return indices, shape


def _oneDimSparsePose(indices, shape):
    ind_onedim = []
    for ind in indices:
        idx = ind[0] * shape[2] * shape[1] + ind[1] * shape[2] + ind[2]
        ind_onedim.append(idx)
    shape = np.prod(shape)
    return ind_onedim, shape


def get_skeleton(peak_points, filename, imsize, params, normalize=None):

    ret = []
    for i in range(len(imsize)):

        peaks = peak_points[filename]
        point_list = list()
        for point in peaks:
            if len(point) > 0:
                x, y, v, _ = point[0]
                point_list.append(x)
                point_list.append(y)
                point_list.append(v)
            else :
                point_list.append(0)
                point_list.append(0)
                point_list.append(0)

        point_list = np.array(point_list)
        skeleton, _ = get_pose((128, 128, 3), point_list)
        skeleton = Image.fromarray(skeleton)

        # padding the image
        w, h = skeleton.size
        skeleton = ImageOps.expand(skeleton,
                                   border=(round((h - w) / 2), 0,
                                           round((h - w) / 2), 0),
                                   fill=(0, 0, 0))  # left,top,right,bottom
        re_skeleton = transforms.Resize(imsize[i], interpolation=Image.BICUBIC)(skeleton)
        re_skeleton = normalize(re_skeleton)

        if params['flip']:
            re_skeleton = torch.flip(re_skeleton, dims=[2])

        ret.append(re_skeleton)

        return ret

def define_edge_lists():
    pose_edge_list = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7],
                      [7, 8], [2, 9], [9, 10],
                      [10, 11], [2, 12], [12, 13],
                      [13, 14], [2, 1], [1, 15], [15, 17],
                      [1, 16], [16, 18], [3, 17], [6, 18]]

    # pose_edge_list = [[2, 3], [2, 6], [3, 4], [4, 5],
    #                   [6, 7], [7, 8], [2, 1], [1, 16],
    #                   [16, 18], [2, 9], [2, 10], [9, 14],
    #                   [14, 15], [10, 11], [11, 12], [1, 17],
    #                   [17, 19]]

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]

    return pose_edge_list, colors

def extract_valid_keypoints(pts):
    p = pts.shape[0]
    thre = 0.1 if p == 70 else 0.01
    output = np.zeros((p, 2))
    valid = (pts[:, 2] > thre)
    output[valid, :] = pts[valid, :2]

    return output

def get_pose(shape, points):
    canvas = np.zeros(shape, dtype=np.uint8)
    keypoints = np.zeros(shape, dtype=np.uint8)
    pose_edge_list, colors = define_edge_lists()

    pose_pts = points.reshape(-1, 3)
    pose_pts = [extract_valid_keypoints(pts) for pts in [pose_pts]]
    pose_pts = pose_pts[0]
    zero_array = np.zeros(((25 -18), 2))
    pose_pts = np.row_stack((pose_pts, zero_array))
    for i, edge in enumerate(pose_edge_list):
        if i > 16:
            break
        edge = [edge[0] - 1, edge[1] - 1]
        joint_coords = pose_pts[edge, :2]

        # Draw circles at every joint
        for joint in joint_coords:
            if 0 in joint[0:2].astype(int):  # 有0的坐标，跳过
                continue
            cv.circle(canvas, tuple(joint[0:2].astype(int)),
                      2, colors[i], thickness=-1)
            cv.circle(keypoints, tuple(joint[0:2].astype(int)),
                      2, colors[i], thickness=-1)

        if 0 in joint_coords[0][0:2].astype(int) \
                or 0 in joint_coords[1][0:2].astype(int):  # 有0的坐标，跳过
            continue

        # Draw line
        coords_center = tuple(
            np.round(np.mean(joint_coords, 0)).astype(int))
        limb_dir = joint_coords[0, :] - joint_coords[1, :]
        limb_length = np.linalg.norm(limb_dir)
        angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        polygon = cv.ellipse2Poly(coords_center, (int(limb_length / 2), 1),
                                  int(angle), 0, 360, 1)
        cv.fillConvexPoly(canvas, polygon, colors[i])

    return canvas, keypoints


def get_heatmaps(peak_points, filename, imsize=None, flip=False):
    heatmaps_list = []
    basic_h, basic_w = 128, 128
    for size in imsize:
        width, height = size, size

        peaks = peak_points[filename]
        print(peaks)
        indices_r4_1, shape = _getSparsePose(
            peaks, basic_h, basic_w, width, height, 18, radius=2, mode='Solid')
        indices_r4_1, shape_1 = _oneDimSparsePose(indices_r4_1, shape)

        indices_r4_1 = np.array(indices_r4_1).astype(np.int64).flatten().tolist()
        indices_r4_1_dense = np.zeros((shape_1))
        indices_r4_1_dense[indices_r4_1] = 1
        indices_r4_1 = np.reshape(indices_r4_1_dense, (height, width, 18))
        pose_1 = indices_r4_1.astype('float32')

        pose_1 = torch.from_numpy(pose_1.transpose(2, 0, 1).astype(np.float32))

        if flip:
            pose_1 = torch.flip(pose_1, dims=[2])

        heatmaps_list.append(pose_1)

    return heatmaps_list


def get_imgs(img_path, imsize, params, normalize=None):
    """
    :param img_path: file path
    :param imsize: [32, 64, 128]
    :param params:
    :param normalize:
    :return:
    """
    img_path = img_path.replace('/img/','/img_x4/')
    img = Image.open(img_path).convert('RGB')

    ret = []
    for i in range(len(imsize)):
        # print(imsize[i])
        re_img = transforms.Resize(imsize[i], interpolation=Image.BICUBIC)(img)
        re_img = normalize(re_img)

        if params['flip']:
            re_img = torch.flip(re_img, dims=[2])

        ret.append(re_img)

    return ret


def get_parsers(parser, imsize, layer=1):

    ret = []
    parser = parser.unsqueeze(0)
    for i in range(len(imsize)):
        re_parser = F.interpolate(parser, imsize[i], mode='nearest')
        # [1, 1, 128, 128]
        if layer == 7:
            w, h = re_parser.shape[2], re_parser.shape[3]
            temp_parser = torch.rand((1, layer, w, h))
            for i in range(layer):
                temp_parser[0, i, :,:] = re_parser[0,0,:,:] == i
            ret.append(temp_parser.squeeze(0))
        else:
            ret.append(re_parser.squeeze(0).squeeze(0))

    return ret


class DeepfashionDataset(data.Dataset):
    def __init__(self, opt, base_size=64, get_num=3,
                 parser_layer=1, word_embedding=None, norm=None):
        super(DeepfashionDataset, self).__init__()

        self.opt = opt
        if norm is not None:
            self.norm = norm
        else:
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = 1  # 每张图片的 caption 数量
        self.parser_layer = parser_layer

        self.word_embedding = word_embedding \
            if word_embedding is not None else None

        self.imsize = []
        if get_num == 1:
            self.imsize.append(256)
        else:
            for i in range(3):
                self.imsize.append(base_size)
                base_size = base_size * 2

        self.data = []
        self.data_dir = opt.data_root_deepfashion

        self.anno = loadmat(os.path.join(
            self.data_dir, 'Anno/language_original.mat'))
        self.index_data = loadmat(os.path.join(self.data_dir, 'Eval/ind.mat'))
        self.img_parser = h5py.File(os.path.join(self.data_dir, 'Img/G2.h5'), 'r')
        key_points_file = os.path.join(self.data_dir, 'peaks.pickle')
        with open(key_points_file, mode='rb') as f:
            self.peaks_dic = pickle.load(f)

        self.index, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(
            self.data_dir, self.anno, self.index_data)

        # print('Get sum of datas: {} !'.format(len(self.index)))
        if opt.control_data_num is not None:
            self.index = self.index[0: opt.control_data_num]
        # print('Lets run datas: {} !'.format(len(self.index)))

        label = []
        for i in range(self.anno['nameList'].shape[0]):
            label.append(str(self.anno['nameList'][:, 0][i][0]).split('/')[1])
        counter = Counter(label)
        self.class_list = list(counter.keys())
        self.number_example = len(self.index)

    def load_captions(self, anno, idx_list):
        all_captions = []
        for idx in idx_list:
            cap = anno['engJ'][:, 0][idx-1][0]
            cap = cap.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(cap.lower())
            # print('tokens', tokens)
            if len(tokens) == 0:
                print('cap', cap)
                continue

            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
            all_captions.append(tokens_new)
        return all_captions

    # if 'captions.pickle' not exists
    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, anno, index_data):
        # filepath = os.path.join(data_dir, 'captions.pickle')
        filepath = '/home/OpenResource/Datasets/MPV/merge_captions.pickle'

        train_idx = list(index_data['train_ind'][:, 0])
        test_idx = list(index_data['test_ind'][:, 0])

        if not os.path.isfile(filepath):
            train_captions = self.load_captions(anno, train_idx)
            test_captions = self.load_captions(anno, test_idx)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            train_captions = self.load_captions(anno, train_idx)
            test_captions = self.load_captions(anno, test_idx)
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                ixtoword, wordtoix = x[1], x[2]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if self.opt.isTrain:
            # a list of list: each list contains the indices of words in a sentence
            captions = train_captions
            idx = train_idx
        else:
            # split=='test' or 'val'
            captions = test_captions
            idx = test_idx
        return idx, captions, ixtoword, wordtoix, n_words

    def __getitem__(self, index):
        params = self._get_param()
        params['flip'] = False
        idx = self.index[index] - 1

        filename = self.anno['nameList'][:, 0][idx][0]
        cls_id = self.class_list.index(str(filename).split('/')[1])

        img_name = os.path.join(self.data_dir, filename)
        imgs = get_imgs(img_name, self.imsize,
                        params, normalize=self.norm)

        parser = self.img_parser['b_'][idx, 0, :, :].T
        parser = torch.from_numpy(parser.astype(np.float32)).unsqueeze(0)
        if params['flip']:
            parser = torch.flip(parser, dims=[2])   # if array --> np.fliplr(parser)
        parsers = get_parsers(parser, self.imsize, layer=self.parser_layer)

        skeleton = get_skeleton(self.peaks_dic, filename.replace('img/', ''),
                                self.imsize, params, normalize=self.norm)

        # heatmaps = get_heatmaps(self.peaks_dic, filename.replace('img/', ''),
        #                         self.imsize, flip=params['flip'])

        sentence = self.anno['engJ'][:, 0][idx][0]
        # caption = sentence.replace('wore', 'is wearing')
        # caption = caption.replace('was', 'is')
        # caption = caption.replace('multicolor', 'multi-color')
        # sentence = caption.replace('multi-colored', 'multi-color')

        caps, cap_len = get_caption(sentence, self.wordtoix,
                                    word_num=18,
                                    word_embedding=self.word_embedding)

        return imgs, parsers, skeleton, caps, cap_len, cls_id, idx, sentence

    @staticmethod
    def _get_param():
        flip = random.random() > 0.5
        return {'flip': flip}

    def __len__(self):
        return len(self.index)


def loader_deepfahsion_data(opt, norm=None):

    data = DeepfashionDataset(opt, base_size=256, get_num=1, norm=norm)
    load_data = DATA.DataLoader(
        data,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.num_works,
        drop_last=True
    )

    return load_data, data


if __name__ == '__main__':
    from easydict import EasyDict as edit
    opt = edit({
        'data_root_deepfashion': '/home/OpenResource/Datasets/DeepFashion/FashionSynthesisBenchmark',
        'batch_size': 1,  # 28,  batch_size 必须大于2
        'isTrain': False,
        'shuffle': True,
        'num_works': 1,
        'control_data_num': 1000,
        'gpu_id': 0
    })

    DEVICE = torch.device('cuda:%d' % opt.gpu_id)
    torch.cuda.set_device(opt.gpu_id)

    from pprint import pprint
    pprint(opt)

    # ========== DATA ============
    dataloader, data = loader_deepfahsion_data(opt)
    # os.makedirs('./check_imgs')
    # from utils import tensor2im, decode_labels, label2onhot
    for step, data in enumerate(dataloader):
        imgs, parsers, skeleton, caps, cap_len, cls_id, idx, sentence = data

        # print(caps)
        # print(cap_len)

        print(imgs[-1].shape)
        print(sentence)

        # print(f'step: {step}, {caps})')
        # print(imgs[-1].shape)
        # # img_vis = tensor2im(imgs[-1])[0]
        # # Image.fromarray(img_vis).save(f'./check_imgs/img_{step}.png')
        # #
        # parser = label2onhot(parsers[-1].unsqueeze(1), layer=7)
        # parser_vis = decode_labels(parser)[0]
        # Image.fromarray(parser_vis).save(f'./check_imgs/parser_{step}.png')
        if step == 1000:
            break

"""
 background, hair, face, upper-clothes, pants,
legs, and arms,
"""