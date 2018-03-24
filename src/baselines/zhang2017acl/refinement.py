from util import get_true_dict, get_embeddings_dicts, get_train_dict
import numpy as np
from properties import *
from sklearn.utils.extmath import randomized_svd
from validation_faiss import get_precision_k, normalize_mat


def training_dataset(train_dict, source_dict, target_dict):
    d = g_input_size
    num_seeds = 5000

    X = np.zeros((d, num_seeds))
    Y = np.zeros((d, num_seeds))

    i = 0
    for source_word, target_words in train_dict.items():
        for target_word in target_words:
            X[:, i] = source_dict[source_word]
            Y[:, i] = target_dict[target_word]
            i += 1

    return X, Y


def get_opt_W(X, Y):
    U, Sigma, VT = randomized_svd(np.matmul(Y, np.transpose(X)),
                                  n_components=np.shape(X)[0])
    return np.matmul(U, VT)


def get_mapped_embeddings(test_dict, W, source_dict):
    test_words = list(test_dict.keys())
    mapped_embeddings = np.zeros((len(test_words), g_input_size))
    for (i, test_word) in enumerate(test_words):
        v = np.array(source_dict[test_word]).astype(float)
        mapped_embeddings[i] = np.matmul(W, v)
    return mapped_embeddings


if __name__ == "__main__":
    k = 1

    train_dict = get_train_dict()
    source_dict, target_dict = get_embeddings_dicts()
    test_dict = get_true_dict()
    target_word_list = list(target_dict.keys())

    X, Y = training_dataset(train_dict, source_dict, target_dict)
    W = get_opt_W(X, Y)
    xq = get_mapped_embeddings(test_dict, W, source_dict)
    print(np.shape(xq))
    xq = normalize_mat(xq)
    print(get_precision_k(k, test_dict, xq, target_word_list))