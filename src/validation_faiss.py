from faiss_master import faiss
import numpy as np
from src.util import get_embeddings_dicts, get_validation_set, \
    get_embeddings, get_true_dict
from src.properties import *
from src.trainer import train, to_tensor, to_variable
import json


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
        # print("%d: %s" % (i, word))
        translation_dict[word] = [target_word_list[j] for j in
                                  list(knn_indices[i])]
    print(json.dumps(translation_dict, indent=2))
    return translation_dict


def get_knn_indices(k, xb, xq):
    index = faiss.IndexFlatL2(g_input_size)
    index.add(xb)
    distances, knn_indices = index.search(xq, k)
    return distances, knn_indices


def get_knn_indices_fast(k, xb, xq):
    nlist = 100
    quantizer = faiss.IndexFlatL2(g_input_size)
    index = faiss.IndexIVFFlat(quantizer, g_input_size, nlist, faiss.METRIC_L2)

    assert not index.is_trained
    index.train(xb)
    assert index.is_trained

    index.add(xb)
    distances, knn_indices = index.search(xq, k)
    return distances, knn_indices


def CSLS(k, xb, xq):
    print("Here-1")
    distances, _ = get_knn_indices(k, xb, xq)
    r_source = np.average(distances, axis=0)
    print("Here-2")
    distances, _ = get_knn_indices(k, xb, xb)
    r_target = np.average(distances, axis=0)
    print("Here-3")

    n_source = np.shape(r_source)[0]
    n_target = np.shape(r_target)[0]
    ones_matrix = np.ones((n_target, g_input_size))
    knn_indices = np.zeros((n_source, k))

    for i in range(n_source):
        print("i: ", i)
        r = np.multiply(r_source[i], ones_matrix)
        m1 = np.multiply(xq[i], ones_matrix)
        c = cosine_similarity(m1, xb)
        csls = 2*c - r - r_target
        k_best_indices = np.argsort(csls)[-1 * k:]
        knn_indices[i] = k_best_indices

    return knn_indices


def CSLS_fast(k, xb, xq):
    distances, _ = get_knn_indices_fast(k, xb, xq)
    r_source = np.average(distances, axis=1)
    distances, _ = get_knn_indices_fast(k, xb, xb)
    r_target = np.average(distances, axis=1)

    n_source = np.shape(r_source)[0]
    print(np.shape(r_source))
    n_target = np.shape(r_target)[0]
    print(np.shape(r_target))
    ones_matrix = np.ones((n_target, g_input_size))
    ones_vector = np.ones((n_target, 1))
    knn_indices = np.zeros((n_source, k))

    for i in range(n_source):
        print("i: ", i)
        r = np.multiply(r_source[i], ones_vector)
        m1 = np.multiply(xq[i], ones_matrix)
        c = cosine_similarity(m1, xb)
        csls = 2*c - r - r_target
        k_best_indices = np.argsort(csls)[-1 * k:]
        knn_indices[i] = k_best_indices

    return knn_indices


def cosine_similarity(m1, m2):
    numerator = np.sum(np.multiply(m1, m2), axis=1)
    denominator = np.sqrt(np.multiply(np.sum(np.multiply(m1, m1), axis=1),
                                      np.sum(np.multiply(m2, m2), axis=1)))
    return numerator/denominator


def get_mapped_embeddings(g, source_word_list):
    source_vec_dict, target_vec_dict = get_embeddings_dicts()
    target_word_list = list(target_vec_dict.keys())
    mapped_embeddings = np.zeros((len(source_word_list), g_input_size))
    for (i, source_word) in enumerate(source_word_list):
        word_tensor = to_tensor(np.array(source_vec_dict[source_word]).astype(float))
        mapped_embeddings[i] = g(to_variable(word_tensor)).data.cpu().numpy()
    return mapped_embeddings, target_word_list


def get_precision_k(k, g, true_dict):
    source_word_list = true_dict.keys()
    _, xb = get_embeddings()
    xb = np.float32(xb)
    xq, target_word_list = get_mapped_embeddings(g, source_word_list)
    xq = np.float32(xq)
    _, knn_indices = get_knn_indices(k, xb, xq)
    predicted_dict = get_translation_dict(source_word_list, target_word_list,
                                          knn_indices)
    return calculate_precision(true_dict, predicted_dict)


def test_function(source_word_list):
    source_vec_dict, _ = get_embeddings_dicts()
    source_embeddings = np.zeros((len(source_word_list), g_input_size))
    for (i, source_word) in enumerate(source_word_list):
        source_embeddings[i] = source_vec_dict[source_word]
    return source_embeddings


if __name__ == '__main__':
    k = 10
    true_dict = get_true_dict()
    source_word_list = true_dict.keys()
    g = train()
    print(get_precision_k(k, g, true_dict))
