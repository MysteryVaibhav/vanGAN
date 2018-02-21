from util import get_word2vec, get_evaluation_data
from model import Generator
from properties import *
import torch
import numpy as np
import queue as Q


# Returns the normalized vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# Returns cosine distance between two vectors
def cosine_dist(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Naive knn implementation, computes distance of word with all vocab words in target language
def knn(en_word, en_w2v, it_w2v, g, k=5):
    neighbours = set()
    if en_word in en_w2v:
        q = Q.PriorityQueue()
        # TODO: Move the below tensor to GPU if needed. Skipping that for now, since the model is also on CPU
        transformed_embedding = g(torch.autograd.Variable(torch.FloatTensor(en_w2v[en_word])))
        transformed_embedding = normalize(transformed_embedding.data.numpy())
        for it_word in it_w2v:
            # TODO: Can be more efficient, embeddings can be pre-normalized
            q.put((-1 * cosine_dist(transformed_embedding, normalize(it_w2v[it_word])), it_word))
        i = 0
        while i < k:
            top = q.get()
            neighbours.add(top[1])
            i += 1
    return neighbours


# Evaluates the bilingual lexicon induction task on EuroParl set
def evaluate(k):
    en_w2v, it_w2v = get_word2vec()
    en_it_mapping_truth = get_evaluation_data()
    g = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    g.load_state_dict(torch.load('generator_weights_1.t7'))
    found = 0
    out_of_vocabulary_count = 0
    for word in en_it_mapping_truth:
        neighbours = knn(word, en_w2v, it_w2v, g, k)
        if len(neighbours) == 0:
            out_of_vocabulary_count += 1
            print("{} not in vocabulary !")
        elif len(neighbours.intersection(en_it_mapping_truth[word])) > 0:
            found += 1
            print("Found !!")
        print(word + ": their neighbours: " + str(neighbours))
    print("Precision at k={} : {}".format(k, found / (len(en_it_mapping_truth) - out_of_vocabulary_count)))


if __name__ == '__main__':
    evaluate(5)
