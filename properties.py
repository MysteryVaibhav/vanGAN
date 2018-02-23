DATA_DIR = 'C:\\Users\\myste\\Downloads\\transmat\\data\\'
EN_WORD_TO_VEC = 'EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
IT_WORD_TO_VEC = 'IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
EVAL_EUROPARL = 'OPUS_en_it_europarl_test.txt'
VALIDATION_FILE = 'OPUS_en_it_europarl_test.txt'

# Model Hyper-Parameters
#TODO: Add all these as program parameters
g_input_size = 300     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 300    # size of generated output vector
d_input_size = 300   # cardinality of distributions
d_hidden_size = 2048   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
mini_batch_size = 128

d_learning_rate = 0.001  # 2e-4
g_learning_rate = 0.001
optim_betas = (0.9, 0.999)
num_epochs = 100
print_interval = 200
d_steps = 5  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1
smoothing = 0.2
beta = 0.0001
