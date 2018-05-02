from os import path

"""Default parameters"""

ROOT_DIR = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
DATA_DIR = path.join(ROOT_DIR, 'data')
if not path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
MODEL_DIR = path.join(ROOT_DIR, 'models')
if not path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
PLOT_DIR = path.join(ROOT_DIR, 'plots')
if not path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

SRC_WORD_TO_VEC = 'wiki.en.subwords.top75000.npz'
TGT_WORD_TO_VEC = 'wiki.ru.words.npz'
VALIDATION_FILE = 'en-ru.5000-6500.subwords'
FULL_FILE = 'en-ru.txt'
NEW_VAL_FILE = 'en-ru-new.txt'
GOLD_FILE = 'en-ru.0-5000.txt'

# For Procrustes (Supervised):
TRAIN_FILE = 'OPUS_en_it_europarl_train_5K.txt'

# Model Hyper-Parameters
g_input_size = 300     # Random noise dimension coming into generator, per output vector
g_output_size = 300    # size of generated output vector
d_input_size = 300   # cardinality of distributions
d_hidden_size = 2048   # Discriminator complexity
g_hidden_size = 300
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
mini_batch_size = 32

d_learning_rate = 0.2
g_learning_rate = 0.2
num_epochs = 100
d_steps = 1
g_steps = 1
smoothing = 0.1   # As per what is mentioned in the paper
beta = 0.001
clip_value = 0

# Training
iters_in_epoch = 100000
most_frequent_sampling_size = 75000
print_every = 1
lr_decay = 1
lr_min = 1e-6
num_random_seeds = 15    # Number of different seeds to try
center_embeddings = 0
k_neighbours_inp = 4

dropout_inp = 0.1
dropout_hidden = 0
leaky_slope = 0.2
add_noise = 0
noise_mean = 1.0
noise_var = 0.2

# Validation
K = 5
top_frequent_words = 200000

# refinement
refine_top = 15000
cosine_top = 10000

# data processing, train or eval
mode = 1

csls_k = 10
dict_max_top = 10000

context = 0
atype = 'dot'
