from os import mkdir
from os import path

CUR_DIR = path.dirname(path.abspath(__file__))
ROOT_DIR = path.dirname(path.dirname(path.dirname(CUR_DIR)))
DATA_DIR = path.join(ROOT_DIR, 'data/')
MODEL_DIR = path.join(ROOT_DIR, 'models')

lang_src = 'en'
lang_trg = 'es'
SRC_WORD_VEC = 'zhang2017acl.en.vec'
TRG_WORD_VEC = 'zhang2017acl.es.vec'
SRC_WORD_FREQ = 'zhang2017acl.en.freq'
TRG_WORD_FREQ = 'zhang2017acl.es.freq'
VALIDATION_FILE = 'en-es.5000-6500.txt'

# Model Hyper-Parameters
g_input_size = 300     # Random noise dimension coming into generator, per output vector
g_output_size = 300    # size of generated output vector
d_input_size = 300   # cardinality of distributions
d_hidden_size = 500   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
mini_batch_size = 128

d_learning_rate = 0.001
g_learning_rate = 0.001
num_epochs = 50
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1
smoothing = 0.1
beta = 0.001
clip_value = 0

# Training
max_iters = 500000
most_frequent_sampling_size = 75000
print_every = 1

# Validation
K = 5
top_frequent_words = 200000
