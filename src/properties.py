DATA_DIR = '/home/ubuntu/vanGAN/data/data/'
EN_WORD_TO_VEC = 'EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
IT_WORD_TO_VEC = 'IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
EVAL_EUROPARL = 'OPUS_en_it_europarl_test.txt'
VALIDATION_FILE = 'OPUS_en_it_europarl_test.txt'
TRAIN_FILE = 'OPUS_en_it_europarl_train_5K.txt'

# Model Hyper-Parameters
g_input_size = 300     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 300    # size of generated output vector
d_input_size = 300   # cardinality of distributions
d_hidden_size = 2048   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
mini_batch_size = 128

d_learning_rate = 0.01
g_learning_rate = 0.01
num_epochs = 20
d_steps = 5  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1
smoothing = 0.1
beta = 0.01
clip_value = 0

# Validation
K = 5