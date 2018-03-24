from os import path

CUR_DIR = path.dirname(path.abspath(__file__))
ROOT_DIR = path.dirname(path.dirname(path.dirname(CUR_DIR)))
DATA_DIR = path.join(ROOT_DIR, 'data/')
EN_WORD_TO_VEC = 'wiki.en.vec' #'EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
IT_WORD_TO_VEC = 'wiki.it.vec' #'IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
VALIDATION_FILE = 'en-it.5000-6500.txt' #'OPUS_en_it_europarl_test.txt'
TRAIN_FILE = 'OPUS_en_it_europarl_train_5K.txt'

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
iters_in_epoch = 1000000
most_frequent_sampling_size = 75000
print_every = 1

# Validation
K = 5
top_frequent_words = 200000
