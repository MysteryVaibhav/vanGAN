import numpy as np
import codecs
from properties import *


# Returns a mapping of words and their embedding
def save_word_vectors(file, dir=DATA_DIR, save_file_as='en'):
    embeddings = []
    keys = []
    with codecs.open(dir + file, 'r', encoding='utf-8', errors='ignore') as f:
        ignore_first_row = True
        for row in f.readlines():
            if ignore_first_row:
                ignore_first_row = False
                continue
            split_row = row.split(" ")
            if len(np.array(split_row[1:])) == 300:
                embeddings.append(np.array(split_row[1:]).astype(np.float))
                keys.append(split_row[0])
    np.save(DATA_DIR + save_file_as + '.npy', np.array(embeddings))
    np.save(DATA_DIR + save_file_as + '_key.npy', np.array(keys))


# Before using these methods make sure you run this util file once to create the data files en.npy and it.npy
# Returns the monolingual embeddings in en and it
def get_embeddings():
    return np.load(DATA_DIR + 'en.npy'), np.load(DATA_DIR + 'it.npy')


# Returns list of words corresponding to the embeddings above
def get_key_embeddings():
    return np.load(DATA_DIR + 'en_key.npy'), np.load(DATA_DIR + 'it_key.npy')


# Combines above two functions to return a mapping of word to embeddings
def get_word2vec():
    en, it = get_embeddings()
    en_key, it_key = get_key_embeddings()
    en_word2vec = {}
    it_word2vec = {}
    assert len(en) == len(en_key) and len(it) == len(it_key)
    for i in range(len(en_key)):
        en_word2vec[en_key[i]] = en[i]
    for i in range(len(it_key)):
        it_word2vec[it_key[i]] = it[i]
    return en_word2vec, it_word2vec


# Returns the mapping of word and its counter-part in other language
def get_evaluation_data(file=EVAL_EUROPARL):
    en_to_it_word_mapping = {}
    with codecs.open(DATA_DIR + file, 'r', encoding='utf-8', errors='ignore') as f:
        for row in f.readlines():
            split_row = row.split(" ")
            if split_row[0] not in en_to_it_word_mapping:
                en_to_it_word_mapping[split_row[0]] = set()
            en_to_it_word_mapping[split_row[0]].add(split_row[1].replace("\n", ""))
    return en_to_it_word_mapping


if __name__ == '__main__':
    # Save embeddings
    save_word_vectors(EN_WORD_TO_VEC, save_file_as='en')
    save_word_vectors(IT_WORD_TO_VEC, save_file_as='it')
