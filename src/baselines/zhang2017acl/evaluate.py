#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.autograd import Variable
import faiss
import numpy as np
import torch

from properties import *
from util import get_embeddings
from util import get_embeddings_dicts
from util import to_tensor

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
    return translation_dict


def get_knn_indices(k, xb, xq):
    index = faiss.IndexFlatIP(g_input_size)
    index.add(xb)
    distances, knn_indices = index.search(xq, k)
    return distances, knn_indices


def CSLS_fast(k, xb, xq):
    distances, _ = get_knn_indices(k, xb, xq)
    r_source = np.average(distances, axis=1)
    distances, _ = get_knn_indices(k, xq, xb)
    r_target = np.average(distances, axis=1)

    n_source = np.shape(r_source)[0]
    n_target = np.shape(r_target)[0]

    knn_indices = []
    for i in range(n_source):
        src_wemb = xq[i, :]
        c = np.sum(np.multiply(np.repeat(src_wemb[np.newaxis, :],  n_target, axis=0), xb), axis=1)
        rs = np.repeat(r_source[i],  n_target, axis=0)
        csls = 2*c - rs - r_target
        knn_indices.append(np.argsort(csls)[-k:])

    return knn_indices


def cosine_similarity(m1, m2):
    numerator = np.sum(np.multiply(m1, m2), axis=1)
    denominator = np.sqrt(np.multiply(np.sum(np.multiply(m1, m1), axis=1),
                                      np.sum(np.multiply(m2, m2), axis=1)))
    return numerator/denominator


def get_mapped_embeddings(g, source_word_list, gpu=False):
    source_vec_dict, target_vec_dict = get_embeddings_dicts()
    target_word_list = list(target_vec_dict.keys())
    mapped_embeddings = np.zeros((len(source_word_list), g_input_size))
    for (i, source_word) in enumerate(source_word_list):
        word_tensor = to_tensor(np.array(source_vec_dict[source_word]).astype(float))
        if gpu:
            word_tensor = word_tensor.cuda()
        mapped_embedding = g(Variable(word_tensor)).data
        if gpu:
            mapped_embedding = mapped_embedding.cpu()
        mapped_embeddings[i] = mapped_embedding.numpy()
    return mapped_embeddings, target_word_list


def get_precision_k(k, g, true_dict, method='csls', gpu=False):
    source_word_list = true_dict.keys()

    _, xb = get_embeddings()
    xb = np.float32(xb)
    row_sum = np.linalg.norm(xb, axis=1)
    xb = xb / row_sum[:, np.newaxis]

    xq, target_word_list = get_mapped_embeddings(g, source_word_list, gpu=gpu)
    xq = np.float32(xq)
    row_sum = np.linalg.norm(xq, axis=1)
    xq = xq / row_sum[:, np.newaxis]

    if method == 'nn':
        _, knn_indices = get_knn_indices(k, xb, xq)
    elif method == 'csls':
        knn_indices = CSLS_fast(k, xb, xq)
    else:
        raise 'Method not implemented: %s' % method

    predicted_dict = get_translation_dict(
        source_word_list, target_word_list, knn_indices)
    return calculate_precision(true_dict, predicted_dict)


def test_function(source_word_list):
    source_vec_dict, _ = get_embeddings_dicts()
    source_embeddings = np.zeros((len(source_word_list), g_input_size))
    for (i, source_word) in enumerate(source_word_list):
        source_embeddings[i] = source_vec_dict[source_word]
    return source_embeddings
