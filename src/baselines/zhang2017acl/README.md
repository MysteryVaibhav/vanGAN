# Implementation of (Zhang et al., ACL2017)

`properties.py`

```python
# Datasets
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
```


## Setting up word vectors

```shell
python main.py --data_dir ../../../data/ --src_file zhang2017acl.es-en.es.vec --src_freq_file zhang2017acl.es-en.es.freq --tgt_file zhang2017acl.es-en.en.vec --tgt_freq_file zhang2017acl.es-en.en.freq --validation_file zhang2017acl.es-en.test.txt --mode 0
```


## Run

```shell
mkdir -p ../../../models/zhang2017acl.es-en/d-hidden-size-500
python main.py --model_dir ../../../models/zhang2017acl.es-en/d-hidden-size-500  --mode 1
```

A model file will be written to `../../../models/zhang2017acl.es-en/d-hidden-size-500`


# MUSE setting (en-es)

`properties.py`

```python
...
# Datasets
EN_WORD_TO_VEC = 'wiki.en.vec'
IT_WORD_TO_VEC = 'wiki.es.vec'
EN_WORD_FREQ = 'wiki.en.freq'
IT_WORD_FREQ = 'wiki.es.freq'
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

# Training
max_iters = 500000
most_frequent_sampling_size = 75000

# Validation
top_frequent_words = 200000
csls_k = 10
```

Run:
```shell
python main.py --mode 0
```

## Run

```shell
mkdir -p ../../../models/wiki.en-es
python main.py --model_dir ../../../models/wiki.en-es  --mode 1
```
