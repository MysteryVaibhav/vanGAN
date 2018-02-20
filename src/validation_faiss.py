from faiss_master import faiss
import numpy as np
from src.util import get_embeddings_dicts, get_validation_set, get_embeddings
from src.properties import *
from src.trainer import train, to_tensor, to_variable


# d = 64                           # dimension
# nb = 100000                      # database size
# nq = 10000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.
#
# index = faiss.IndexFlatL2(d)   # build the index
# print(index.is_trained)
# index.add(xb)                  # add vectors to the index
# print(index.ntotal)
#
# k = 4                          # we want to see 4 nearest neighbors
# D, I = index.search(xb[:5], k) # sanity check
# print(I)
# print(D)
# D, I = index.search(xqu=-memoryview, k)     # actual search
# print(np.shape(I))                    # neighbors of the 5 first queries

def calculate_precision(true_dict, predicted_dict):
    """Calculates precision given true and predicted dictionaries
    Input:
        true_dict - true dictionary of words to possible translations
        predicted_dict - predicted dictionary of words to possible translations
    Output:
        Precision value
    """
    total_correct = 0
    for (word, translation) in predicted_dict.items():
        true_translations = set(true_dict[word])
        predicted_translations = set(translation)
        if len(true_translations.intersection(predicted_translations)) > 0:
            total_correct += 1
    return float(total_correct)/len(predicted_dict.keys())


def get_translation_dict(source_word_list, target_word_list, knn_indices):
    translation_dict = {}
    for (i, word) in enumerate(source_word_list):
        print("%d: %s" % (i, word))
        translation_dict[word] = [target_word_list[j] for j in
                                  list(knn_indices[i])]
    print(translation_dict)
    return translation_dict


def get_knn_indices(k, xb, xq):
    index = faiss.IndexFlatL2(g_input_size)
    index.add(xb)
    _, knn_indices = index.search(xq, k)
    return knn_indices


def get_mapped_embeddings(g, source_word_list):
    source_vec_dict, target_vec_dict = get_embeddings_dicts()
    target_word_list = list(target_vec_dict.keys())
    mapped_embeddings = np.zeros((len(source_word_list), g_input_size))
    for (i, source_word) in enumerate(source_word_list):
        word_tensor = to_tensor(np.array(source_vec_dict[source_word]).astype(float))
        mapped_embeddings[i] = g(to_variable(word_tensor)).data.numpy()
    return mapped_embeddings, target_word_list


def get_precision_k(k, g, source_word_list):
    _, xb = get_embeddings()
    xb = np.float32(xb)
    xq, target_word_list = get_mapped_embeddings(g, source_word_list)
    xq = np.float32(xq)
    knn_indices = get_knn_indices(k, xb, xq)
    predicted_dict = get_translation_dict(source_word_list, target_word_list,
                                          knn_indices)
    return calculate_precision(true_dict, predicted_dict)


if __name__ == '__main__':
    k = 5
    true_dict = get_validation_set(VALIDATION_FILE, dir=DATA_DIR, save=False)
    source_word_list = true_dict.keys()
    g = train()
    print(get_precision_k(k, g, source_word_list))
