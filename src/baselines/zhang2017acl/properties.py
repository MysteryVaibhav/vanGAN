from os import mkdir
from os import path

CUR_DIR = path.dirname(path.abspath(__file__))
ROOT_DIR = path.dirname(path.dirname(path.dirname(CUR_DIR)))
DATA_DIR = path.join(ROOT_DIR, 'data/')
MODEL_DIR = path.join(ROOT_DIR, 'models')

# See ../../../data/README.md for setting up
EN_WORD_TO_VEC = 'zhang2017acl.es-en.es.vec'
IT_WORD_TO_VEC = 'zhang2017acl.es-en.en.vec'
EN_WORD_FREQ = 'zhang2017acl.es-en.es.freq'
IT_WORD_FREQ = 'zhang2017acl.es-en.en.freq'
VALIDATION_FILE = 'zhang2017acl.es-en.test.txt'

# Model Hyper-Parameters
g_input_size = 50     # Random noise dimension coming into generator, per output vector
g_output_size = 50    # size of generated output vector
d_input_size = 50   # cardinality of distributions
d_hidden_size = 500   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
mini_batch_size = 128

d_learning_rate = 0.001
g_learning_rate = 0.001

# Training
max_iters = 500000
most_frequent_sampling_size = 75000

# Validation
top_frequent_words = 200000
csls_k = 10
