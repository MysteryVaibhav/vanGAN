import numpy as np
import codecs
from src.properties import *


# Returns a mapping of words and their embedding
def get_word_vectors(file, dir=DATA_DIR, save=False, save_file_as='en'):
    word2vec = {}
    embeddings = []
    with codecs.open(dir + file, 'r', encoding='utf-8', errors='ignore') as f:
        ignore_first_row = True
        for row in f.readlines():
            if ignore_first_row:
                ignore_first_row = False
                continue
            split_row = row.split(" ")
            word2vec[split_row[0]] = np.array(split_row[1:]).astype(np.float)
            if save and len(word2vec[split_row[0]]) == 300:
                embeddings.append(word2vec[split_row[0]])
    if save:
        np.save(DATA_DIR + save_file_as + '.npy', np.array(embeddings))
    return word2vec


def get_word_vectors_dicts(file, dir=DATA_DIR, save=False,
                           save_file_as='en_dict'):
    word2vec = {}
    with codecs.open(dir + file, 'r', encoding='utf-8', errors='ignore') as f:
        ignore_first_row = True
        for row in f.readlines():
            if ignore_first_row:
                ignore_first_row = False
                continue
            split_row = row.split(" ")
            vec = np.array(split_row[1:]).astype(np.float)
            if len(vec) == 300:
                word2vec[split_row[0]] = vec
    if save:
        np.save(dir + save_file_as + '.npy', word2vec)
    return word2vec


def get_validation_set(file, dir=DATA_DIR, save=False,
                       save_file_as='validation'):
    true_dict = {}
    with open(dir + file, 'r', encoding='utf-8', errors='ignore') as f:
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


# Before using this method make sure you run this util file once to create the data files en.npy and it.npy
# Returns the monolingual embeddings in en and it
def get_embeddings():
    return np.load(DATA_DIR + 'en.npy'), np.load(DATA_DIR + 'it.npy')


def get_embeddings_dicts():
    return np.load(DATA_DIR + 'en_dict.npy').item(), np.load(DATA_DIR +
                                                        'it_dict.npy').item()


def get_true_dict():
    return np.load(DATA_DIR + 'validation.npy').item()


if __name__ == '__main__':
    # Sanity check
    # word2vec_en = get_word_vectors(EN_WORD_TO_VEC, save=True, save_file_as='en')
    # word2vec_it = get_word_vectors(IT_WORD_TO_VEC, save=True, save_file_as='it')
    # print(word2vec_en['document'])

    # word2vec_en = get_word_vectors_dicts(EN_WORD_TO_VEC, save=True,
    #                                save_file_as='en_dict')
    # word2vec_it = get_word_vectors_dicts(IT_WORD_TO_VEC, save=True,
    #                                save_file_as='it_dict')
    # print(word2vec_en['document'])

    true_dict = get_validation_set(VALIDATION_FILE, save=True)
    print(true_dict)
    print(len(true_dict.keys()))