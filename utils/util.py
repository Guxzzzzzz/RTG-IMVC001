import numpy as np
import torch
from torch.autograd import Variable
import logging
import datetime
import os
import random
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def _get_legacy_mask(data_len, missing_rate=0.3, view_num=2):
    """Randomly generate incomplete data information, simulate partial view data with complete view data"""
    full_matrix = np.ones((int(data_len * (1 - missing_rate)), view_num))
    alldata_len = data_len - int(data_len * (1 - missing_rate))
    if alldata_len != 0:
        one_rate = 1.0 - missing_rate
        if one_rate <= (1 / view_num):
            enc = OneHotEncoder()  # n_values=view_num
            view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
            full_matrix = np.concatenate([view_preserve, full_matrix], axis=0)
            choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
            matrix = full_matrix[choice]
            return matrix
        error = 1
        if one_rate == 1:
            matrix = randint(1, 2, size=(alldata_len, view_num))
            full_matrix = np.concatenate([matrix, full_matrix], axis=0)
            choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
            matrix = full_matrix[choice]
            return matrix
        while error >= 0.005:
            enc = OneHotEncoder()  # n_values=view_num
            view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
            one_num = view_num * alldata_len * one_rate - alldata_len
            ratio = one_num / (view_num * alldata_len)
            matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
            a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))
            one_num_iter = one_num / (1 - a / one_num)
            ratio = one_num_iter / (view_num * alldata_len)
            matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
            matrix = ((matrix_iter + view_preserve) > 0).astype(np.int_)
            ratio = np.sum(matrix) / (view_num * alldata_len)
            error = abs(one_rate - ratio)
        full_matrix = np.concatenate([matrix, full_matrix], axis=0)

    choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
    matrix = full_matrix[choice]
    return matrix


def get_mask(data_len, missing_rate=0.3, view_num=2,
             protocol='incomplete_sample'):
    """Generate a reproducible partial-view mask.

    Under the formal ``incomplete_sample`` protocol, ``missing_rate`` is the
    fraction of samples that retain exactly one randomly selected view.  The
    remaining samples retain every view.  This definition is continuous and
    directly auditable at 0.3/0.5/0.7/0.9.  ``legacy`` preserves the original
    entry-rate heuristic only for historical reproduction.
    """
    data_len = int(data_len)
    view_num = int(view_num)
    missing_rate = float(missing_rate)
    if data_len < 1:
        raise ValueError('data_len must be positive')
    if view_num < 2:
        raise ValueError('view_num must be at least two')
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError('missing_rate must lie in [0, 1)')
    if protocol == 'legacy':
        return _get_legacy_mask(data_len, missing_rate, view_num)
    if protocol != 'incomplete_sample':
        raise ValueError(
            "mask protocol must be 'incomplete_sample' or 'legacy'"
        )

    incomplete_count = int(round(data_len * missing_rate))
    complete_count = data_len - incomplete_count
    complete = np.ones((complete_count, view_num), dtype=np.int64)
    incomplete = np.zeros((incomplete_count, view_num), dtype=np.int64)
    if incomplete_count:
        # Balance which view is retained before shuffling.  This prevents the
        # missing-view count itself from becoming a high-variance confounder.
        retained_views = np.arange(incomplete_count) % view_num
        np.random.shuffle(retained_views)
        incomplete[np.arange(incomplete_count), retained_views] = 1
    matrix = np.concatenate((complete, incomplete), axis=0)
    choice = np.random.permutation(data_len)
    return matrix[choice]


def target_l2(q):
    return ((q ** 2).t() / (q ** 2).sum(1)).t()


def normalize(x):
    """ Normalize """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def get_logger(config, main_dir='./logs/'):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    plt_name = str(config['dataset']) + ' ' + str(config['missing_rate']).replace('.', ' ') + ' ' + str(
        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S'))

    fh = logging.FileHandler(
        main_dir + str(config['dataset']) + ' ' + str(config['missing_rate']).replace('.', '') + ' ' + str(
            datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')) + '.logs')

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, plt_name


def get_device():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str('0')  # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    return device


def setup_seed(seed):
    """set up random seed"""
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True
