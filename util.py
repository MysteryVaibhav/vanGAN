import numpy as np
import codecs
from properties import *


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


# Before using this method make sure you run this util file once to create the data files en.npy and it.npy
# Returns the monolingual embeddings in en and it
def get_embeddings():
    return np.load(DATA_DIR + 'en.npy'), np.load(DATA_DIR + 'it.npy')


if __name__ == '__main__':
    # Sanity check
    word2vec_en = get_word_vectors(EN_WORD_TO_VEC, save=True, save_file_as='en')
    word2vec_it = get_word_vectors(IT_WORD_TO_VEC, save=True, save_file_as='it')
    print(word2vec_en['document'])