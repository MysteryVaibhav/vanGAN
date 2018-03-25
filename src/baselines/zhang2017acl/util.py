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
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)


# Returns a mapping of words and their embedding
def get_word_vectors(path_embs, path_freqs=None, dirname=DATA_DIR,
                     save=False, save_file_as=lang_src):
    word2freq = defaultdict(int)
    if path_freqs:
        with open(path.join(dirname, path_freqs), 'r', encoding='utf-8') as f:
            for line in f:
                word, freq = line.strip().split(' ')
                word2freq[word] = int(freq)

    embeddings = []
    freqs = []
    print('Read ' + path.join(dirname, path_embs))
    with open(path.join(dirname, path_embs), 'r', encoding='utf-8') as f:
        N, D = f.readline().strip().split()
        D = int(D)
        next(f)  # Skip first row
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
        np.save(DATA_DIR + save_file_as + '.npy', np.array(embeddings))
        np.save(DATA_DIR + save_file_as + '.freq.npy', np.array(freqs))
    return np.array(embeddings), freqs


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


def get_embeddings(lang_src=lang_src, lang_trg=lang_trg, normalize=True):
    src = np.load(path.join(DATA_DIR, lang_src + '.npy'))
    trg = np.load(path.join(DATA_DIR, lang_trg + '.npy'))
    if normalize:
        src = src / np.linalg.norm(src, axis=1).reshape((-1, 1))
        trg = trg / np.linalg.norm(trg, axis=1).reshape((-1, 1))
    return src, trg


def get_frequencies(lang_src=lang_src, lang_trg=lang_trg):
    src = np.load(path.join(DATA_DIR, lang_src + '.freq.npy'))
    trg = np.load(path.join(DATA_DIR, lang_trg + '.freq.npy'))
    return src, trg


def get_embeddings_dicts(lang_src=lang_src, lang_trg=lang_trg):
    dict_src = np.load(DATA_DIR + lang_src + '_dict.npy').item()
    dict_trg = np.load(DATA_DIR + lang_trg + '_dict.npy').item()
    return dict_src, dict_trg


def get_true_dict():
    return np.load(DATA_DIR + 'validation.npy').item()


if __name__ == '__main__':
    print('Reading english word embeddings...')
    word_vec_src, _ = get_word_vectors(SRC_WORD_VEC, SRC_WORD_FREQ,
                                       save=True, save_file_as=lang_src)
    print(word_vec_src.shape)

    print('Reading trgalian word embeddings...')
    word_vec_trg, _ = get_word_vectors(TRG_WORD_VEC, TRG_WORD_FREQ,
                                       save=True, save_file_as=lang_trg)
    print(word_vec_trg.shape)

    print('Creating word vectors for both languages...')
    word_vec_src = get_word_vectors_dicts(SRC_WORD_VEC, save=True,
                                    save_file_as=lang_src + '_dict')
    word_vec_trg = get_word_vectors_dicts(TRG_WORD_VEC, save=True,
                                    save_file_as=lang_trg + '_dict')

    print('Reading validation file...')
    true_dict = get_validation_set(VALIDATION_FILE, save=True)
    
    print('Done !!')


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
