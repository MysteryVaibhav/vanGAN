import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/')
EN_WORD_TO_VEC = 'wiki.en.vec' #'EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
IT_WORD_TO_VEC = 'wiki.it.vec' #'IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
VALIDATION_FILE = 'en-it.5000-6500.txt' #'OPUS_en_it_europarl_test.txt'
TRAIN_FILE = 'OPUS_en_it_europarl_train_5K.txt'

# Model Hyper-Parameters
g_input_size = 300     # Random noise dimension coming into generator, per output vector
g_output_size = 300    # size of generated output vector
d_input_size = 300   # cardinality of distributions
d_hidden_size = 2048   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
mini_batch_size = 32

d_learning_rate = 0.1
g_learning_rate = 0.1
num_epochs = 10
d_steps = 5  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1
smoothing = 0.1
beta = 0.001
clip_value = 0
csls_k = 10

# Training
iters_in_epoch = 10000
most_frequent_sampling_size = 75000
print_every = 1

# Validation
K = 5
top_frequent_words = 200000

# refinement
top_refine = 15000