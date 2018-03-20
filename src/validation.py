from src.util import get_embeddings_dicts, get_validation_set
import numpy as np
from src.trainer import train, to_tensor, to_variable
from src.properties import *


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


def get_best_k(k, source_words, mapped_embeddings):
    translation_dict = {}
    for (i, word) in enumerate(source_words):
        print("%d: %s" % (i, word))
        translation_dict[word] = get_best_k_for_word(k, word, mapped_embeddings)
    return translation_dict


def get_best_k_for_word(k, word, en):
    _, it = get_embeddings_dicts()
    en_word_vec = en[word]
    cos_sim = []
    it_words = list(it.keys())
    for it_word in it_words:
        it_word_vec = it[it_word]
        cos_sim.append(np.dot(en_word_vec, it_word_vec)/(np.linalg.norm(
            en_word_vec) * np.linalg.norm(it_word_vec)))

    k_best_translations = []
    k_best_indices = np.argsort(cos_sim)[-1*k:]
    for i in k_best_indices:
        k_best_translations.append(it_words[i])

    return k_best_translations


def get_mapped_embeddings(g):
    en, _ = get_embeddings_dicts()
    en_words = list(en.keys())
    mapped_embeddings = {}
    for en_word in en_words:
        word_tensor = to_tensor(np.array(en[en_word]).astype(float))
        mapped_embeddings[en_word] = g(to_variable(word_tensor)).data.numpy()
    return en, mapped_embeddings


def get_precision_k(k, true_dict, mapped_embeddings):
    source_words = true_dict.keys()
    predicted_dict = get_best_k(k, source_words, mapped_embeddings)
    return calculate_precision(true_dict, predicted_dict)


if __name__ == '__main__':
    k = 10
    true_dict = get_validation_set(VALIDATION_FILE, dir=DATA_DIR, save=False)
    g = train()
    en, mapped_embeddings = get_mapped_embeddings(g)
    get_precision_k(k, true_dict, mapped_embeddings)