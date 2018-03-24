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
def get_word_vectors(file, dir=DATA_DIR, save=False, save_file_as='en'):
    embeddings = []
    keys = []
    count = 0
    with open(dir + file, 'r', encoding='utf-8') as f:
        next(f)  # Skip first row
        for count, row in enumerate(f, start=1):
            split_row = row.split(" ")
            vec = np.array(split_row[1:-1]).astype(np.float)
            if len(vec) == 300:
                embeddings.append(vec)
                keys.append(split_row[0])
            if count == top_frequent_words:
                break
    np.save(DATA_DIR + save_file_as + '.npy', np.array(embeddings))
    return np.array(embeddings)


def get_word_vectors_dicts(file, dir=DATA_DIR, save=False,
                           save_file_as='en_dict'):
    word2vec = {}
    count = 0
    with open(dir + file, 'r', encoding='utf-8') as f:
        ignore_first_row = True
        for row in f.readlines():
            if ignore_first_row:
                ignore_first_row = False
                continue
            split_row = row.split(" ")
            vec = np.array(split_row[1:-1]).astype(np.float)
            if len(vec) == 300:
                word2vec[split_row[0]] = vec
            count += 1
            if count == top_frequent_words:
                break
    if save:
        np.save(dir + save_file_as + '.npy', word2vec)
    return word2vec


def get_validation_set(file, dir=DATA_DIR, save=False, save_file_as='validation'):
    true_dict = {}
    with open(dir + file, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for row in rows:
            split_row = row.split(" ")
            key = split_row[0]
            value = split_row[1].rstrip("\n")
            if key not in true_dict.keys():
                true_dict[key] = []
                true_dict[split_row[0]].append(value)
    if save:
        np.save(dir + save_file_as + '.npy', true_dict)
    return true_dict


def get_embeddings(lang_src='en', lang_trg='it', normalize=True):
    en = np.load(path.join(DATA_DIR, lang_src + '.npy'))
    it = np.load(path.join(DATA_DIR, lang_trg + '.npy'))
    if normalize:
        en = en / np.linalg.norm(en, axis=1).reshape((-1, 1))
        it = it / np.linalg.norm(it, axis=1).reshape((-1, 1))
    return en, it


def get_embeddings_dicts():
    return np.load(DATA_DIR + 'en_dict.npy').item(), np.load(DATA_DIR + 'it_dict.npy').item()


def get_true_dict():
    return np.load(DATA_DIR + 'validation.npy').item()


if __name__ == '__main__':
    print("Reading english word embeddings...")
    word2vec_en = get_word_vectors(EN_WORD_TO_VEC, save=True, save_file_as='en')
    print(word2vec_en.shape)

    print("Reading italian word embeddings...")
    word2vec_it = get_word_vectors(IT_WORD_TO_VEC, save=True, save_file_as='it')
    print(word2vec_it.shape)

    print("Creating word vectors for both languages...")
    word2vec_en = get_word_vectors_dicts(EN_WORD_TO_VEC, save=True,
                                    save_file_as='en_dict')
    word2vec_it = get_word_vectors_dicts(IT_WORD_TO_VEC, save=True,
                                    save_file_as='it_dict')
    
    print("Reading validation file...")
    true_dict = get_validation_set(VALIDATION_FILE, save=True)
    
    print("Done !!")
