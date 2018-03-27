from collections import defaultdict
from collections import OrderedDict
from os import path
import logging
import numpy as np
import torch

from properties import *


def init_logger(name='logger'):
    """Initialize and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def to_tensor(numpy_array):
    tensor = torch.from_numpy(numpy_array).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def to_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)


def get_word_vectors_dicts(filename, dirname=DATA_DIR, save=False,
                           save_file_as='en_dict'):
    word2vec = OrderedDict()
    count = 0
    print('Read ' + path.join(dirname, filename))
    with open(path.join(dirname, filename), 'r', encoding='utf-8') as f:
        N, D = f.readline().strip().split()
        D = int(D)
        for count, line in enumerate(f.readlines()):
            row = line.strip().split(' ')
            word = row[0]
            vec = np.array(row[1:]).astype(np.float)
            if len(vec) == D:
                word2vec[word] = vec
            count += 1
            if count == top_frequent_words:
                break
    if save:
        np.save(path.join(dirname, save_file_as + '.npy'), word2vec)
    return word2vec


def get_validation_set(filename, dirname=DATA_DIR, save=False, save_file_as='validation'):
    true_dict = OrderedDict()
    print('Read ' + path.join(dirname, filename))
    with open(path.join(dirname, filename), 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for row in rows:
            split_row = row.split(" ")
            key = split_row[0]
            value = split_row[1].rstrip("\n")
            if key not in true_dict.keys():
                true_dict[key] = []
                true_dict[split_row[0]].append(value)
    if save:
        np.save(path.join(dirname, save_file_as + '.npy'), true_dict)
    return true_dict


def get_frequencies():
    src = np.load(path.join(DATA_DIR, 'src.freq.npy'))
    trg = np.load(path.join(DATA_DIR, 'tgt.freq.npy'))
    return src, trg


class WeightedSampler():
    def __init__(self, weights, replacement=True):
        """Initialize a sampler.

        Arguments:
        - weights (list)   : a list of weights, not necessary summing up to one
        - replacement (bool): if ``True``, samples are drawn with replacement.
        """
        self.weights = torch.DoubleTensor(weights)
        self.replacement = replacement

    def get_iterator(self, batch_size):
        """Generate a batch of samples infinitely."""
        while True:
            yield torch.multinomial(self.weights, batch_size, self.replacement)
        

def downsample_frequent_words(counts, thresh=1e-3):
    """Discount frequent words."""
    total_count = counts.sum()
    indices_zero = counts == 0
    counts[indices_zero] = 1.0  # avoid a numerical error
    threshold_count = float(thresh * total_count)
    new_weights = (np.sqrt(counts / threshold_count) + 1) * (threshold_count / counts)
    new_weights = np.maximum(new_weights, 1.0)
    new_weights *= counts
    new_weights[indices_zero] = 0.0
    return new_weights / new_weights.sum()

class Utils:
    def __init__(self, params):
        self.data_dir = params.data_dir
        self.src_file = params.src_file
        self.tgt_file = params.tgt_file
        self.src_freq_file = params.src_freq_file
        self.tgt_freq_file = params.tgt_freq_file
        self.validation_file = params.validation_file
        self.top_frequent_words = params.top_frequent_words

    def run(self):
        print("Reading source word embeddings...")
        word2vec_src, _ = self.save_word_vectors(self.src_file, self.src_freq_file,
                                                 save=True, save_file_as='src')
        print("Done.")
        print(word2vec_src.shape)
        print("Reading target word embeddings...")
        word2vec_tgt, _ = self.save_word_vectors(self.tgt_file, self.tgt_freq_file,
                                                 save=True, save_file_as='tgt')
        print("Done.")
        print(word2vec_tgt.shape)
        print("Reading validation file...")
        self.save_validation_set(self.validation_file, save=True)
        print("Done.")

        print("Constructing source word-id map...")
        self.save_word_ids_dicts(self.src_file, save=True, save_file_as='src_ids')
        print("Done.")
        print("Constructing target word-id map...")
        self.save_word_ids_dicts(self.tgt_file, save=True, save_file_as='tgt_ids')
        print("Everything Done.")

    def save_word_vectors(self, emb_file, freq_file=None, save=False, save_file_as='src'):
        word2freq = defaultdict(int)
        if freq_file:
            with open(path.join(self.data_dir, freq_file), 'r', encoding='utf-8') as f:
                for line in f:
                    word, freq = line.strip().split(' ')
                    word2freq[word] = int(freq)

        embeddings, freqs = [], []
        print('Read ' + path.join(self.data_dir, emb_file))
        with open(path.join(self.data_dir, emb_file), 'r', encoding='utf-8') as f:
            N, D = f.readline().strip().split()
            D = int(D)
            for count, line in enumerate(f, start=1):
                row = line.strip().split(' ')
                word = row[0]
                vec = np.array(row[1:]).astype(np.float)
                if len(vec) == D:
                    embeddings.append(vec)
                    freqs.append(word2freq[word])
                if count == top_frequent_words:
                    break
            if save:
                np.save(path.join(self.data_dir, save_file_as + '.npy'),
                        np.array(embeddings))
                np.save(path.join(self.data_dir, save_file_as + '.freq.npy'),
                        np.array(freqs))
        return np.array(embeddings), freqs

    def save_word_ids_dicts(self, file, save=False, save_file_as='src_ids'):
        word2id = {}
        count = 0
        with open(path.join(self.data_dir, file), 'r', encoding='utf-8') as f:
            _, dim = f.readline().strip().split()
            dim = int(dim)
            for row in f:
                split_row = row.strip().split(" ")
                vec = np.array(split_row[1:]).astype(np.float)
                if len(vec) == dim:
                    word2id[split_row[0]] = count
                count += 1
                if count == self.top_frequent_words:
                    break
        if save:
            np.save(self.data_dir + save_file_as + '.npy', word2id)
        return word2id

    def save_validation_set(self, file, save=False, save_file_as='validation'):
        true_dict = {}
        with open(self.data_dir + file, 'r', encoding='utf-8') as f:
            rows = f.readlines()
            for row in rows:
                split_row = row.split(" ")
                key = split_row[0]
                value = split_row[1].rstrip("\n")
                if key not in true_dict.keys():
                    true_dict[key] = []
                true_dict[split_row[0]].append(value)
        if save:
            np.save(self.data_dir + save_file_as + '.npy', true_dict)
        return true_dict


def load_npy_one(data_dir, fname):
    return np.load(data_dir + fname).item()


def load_npy_two(data_dir, src_fname, tgt_fname, dict=False):
    if dict:
        x = np.load(data_dir + src_fname).item()
        y = np.load(data_dir + tgt_fname).item()
    else:
        x = np.load(data_dir + src_fname)
        y = np.load(data_dir + tgt_fname)
    return x, y


# Validation set in a dictionary form {src_wrd: [tgt_wrd_1, tgt_wrd_2, ...]}
def get_validation_set_ids(data_dir, validation_fname='validation.npy'):
    val_dict = load_npy_one(data_dir, validation_fname)
    src_ids, tgt_ids = load_npy_two(data_dir, 'src_ids.npy', 'tgt_ids.npy', dict=True)
    val_dict_ids = {}
    for src_wrd, tgt_list in val_dict.items():
        val_dict_ids[src_ids[src_wrd]] = [tgt_ids[tgt_wrd] for tgt_wrd in tgt_list
                                          if tgt_wrd in tgt_ids]
    return val_dict_ids


def convert_to_embeddings(emb_array):
    emb_tensor = to_tensor(emb_array)
    v, d = emb_tensor.size()
    emb = torch.nn.Embedding(v, d)
    if torch.cuda.is_available():
        emb = emb.cuda()
    emb.weight.data.copy_(emb_tensor)
    emb.weight.requires_grad = False
    return emb
