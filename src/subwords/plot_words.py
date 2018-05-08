from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from properties import *
import numpy as np
from model import Generator
import torch

# high frequency words
frequent_en = [
   'seven',
   'american',
   'office',
   'template',
]

frequent_es = [
   'siete',
   'americano',
   'oficinas',
   'plantilla',
]

less_frequent_en = [
    'goatherd',
    'fixity',
    'unreasonable',
    'harrowing'
]

less_frequent_es = [
    'cabrero',
    'fijeza',
    'irracional',
    'desgarradora'
]


def plot(concept_words, X, fig, col, size):
    df = pd.DataFrame(X, index=concept_words, columns=['x', 'y'])
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'], s=size, c=col)
    for word, pos in df.iterrows():
        ax.annotate(word, pos, color=col)


def reduce_dimensions_and_plot(word2vec_en, word2vec_en_ids, word2vec_es, word2vec_es_ids, mat):
    n = 2000
    # Taking top n most frequent words for the purpose of plotting
    vecs = []
    for i in range(n):
        vecs.append(word2vec_en[i])
    vecs += [word2vec_en[word2vec_en_ids[word]] for word in frequent_en]
    vecs += [word2vec_en[word2vec_en_ids[word]] for word in less_frequent_en]
        
    # Doing dimensionality reduction for plotting
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(vecs)

    
    # Plot figure
    fig = plt.figure()
    plot(frequent_en, X_tsne[n:n+4], fig, 'red', 20)
    fig.savefig(PLOT_DIR + "word_plot_en_frequent_before_rotation.png")
    
    # Plot figure
    fig = plt.figure()
    plot(less_frequent_en, X_tsne[n+4:], fig, 'red', 20)
    fig.savefig(PLOT_DIR + "word_plot_en_less_frequent_before_rotation.png")

    
    vecs = []
    for i in range(n):
        vecs.append(word2vec_es[i])
    vecs += [word2vec_es[word2vec_es_ids[word]] for word in frequent_es]
    vecs += [word2vec_es[word2vec_es_ids[word]] for word in less_frequent_es]
    vecs += [np.matmul(mat, word2vec_en[word2vec_en_ids[word]]) for word in frequent_en]
    vecs += [np.matmul(mat, word2vec_en[word2vec_en_ids[word]]) for word in less_frequent_en]
    
    # Doing dimensionality reduction for plotting
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(vecs)

    
    # Plot figure
    fig = plt.figure()
    plot(frequent_es, X_tsne[n:n + 4], fig, 'blue', 20)
    fig.savefig(PLOT_DIR + "word_plot_frequent_es.png")
    
    # Plot figure
    fig = plt.figure()
    plot(less_frequent_es, X_tsne[n + 4:n + 8], fig, 'blue', 20)
    fig.savefig(PLOT_DIR + "word_plot_less_frequent_es.png")

    
    fig = plt.figure()
    plot(frequent_es, X_tsne[n:n + 4], fig, 'blue', 20)
    plot(frequent_en, X_tsne[n + 8:n + 12], fig, 'red', 20)
    fig.savefig(PLOT_DIR + "word_plot_en_frequent_after_rotation.png")
    
    fig = plt.figure()
    plot(less_frequent_es, X_tsne[n + 4:n + 8], fig, 'blue', 20)
    plot(less_frequent_en, X_tsne[n + 12:], fig, 'red', 20)
    fig.savefig(PLOT_DIR + "word_plot_en_less_frequent_after_rotation.png")


if __name__ == '__main__':
    g = Generator(input_size=g_input_size, output_size=g_output_size)
    # Generator weights to load transformation matrix
    g.load_state_dict(
        torch.load(MODEL_DIR + 'generator_weights_en_es_seed_13_75000_0.2_67.866.t7', map_location=lambda storage, loc: storage))

    mat = g.map1.weight.data.numpy()

    word2vec_en = np.load(DATA_DIR + 'src.npy')
    word2vec_en_ids = np.load(DATA_DIR + 'src_ids.npy').item()

    word2vec_es = np.load(DATA_DIR + 'tgt.npy')
    word2vec_es_ids = np.load(DATA_DIR + 'tgt_ids.npy').item()

    print("Rank of words by frequency: en:es")
    print([word2vec_en_ids[word] for word in frequent_en])
    print([word2vec_es_ids[word] for word in frequent_es])

    reduce_dimensions_and_plot(word2vec_en, word2vec_en_ids, word2vec_es, word2vec_es_ids, mat)

