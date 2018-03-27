# vanGAN
11-747 Course Project: GAN implementation for Bilingual Lexicon Induction

Requirements for compiling:
------------------------------
- If using wackyDataset:
    - Download data from http://clic.cimec.unitn.it/~georgiana.dinu/down/
    - Make sure you have these 3 files in the data directory :
        - EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt
        - IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt
        - OPUS_en_it_europarl_test.txt
- If using wiki fastText embeddings:
    - Download data in the 'data' directory:
    ```
    # Source fastText Wikipedia embeddings (~6G) (Example: en)
    curl -Lo wiki.en.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
    # Target fastText Wikipedia embeddings (~2G) (Example: it)
    curl -Lo wiki.it.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.it.vec
    # Test file containing translations for 1500 en words (Example: en-it)
    wget https://s3.amazonaws.com/arrival/dictionaries/en-it.5000-6500.txt
    ```
    - Monolingual corpora for other languages can be found at: https://github.com/facebookresearch/fastText/
    - Bilingual corpora for other languages can be found at: https://github.com/facebookresearch/MUSE
- Copy new validation file (example: en-it-new.txt), gold file (example: en-it.0-5000.txt) from new_val_sets/ and bilingual_dicts/ directories (respectively) into data/ folder.
-
- Follow the steps mentioned below to install Faiss (library for efficient similarity search on GPU)
  or simply use "conda install faiss-gpu -c pytorch" [Note conda install will only work with cuda >= 9.0]

Script to compile:
------------------------------
First run main.py in mode-0 to create the .npy files for embeddings.
```
$ python main.py --mode 0
Failed to load GPU Faiss: No module named 'swigfaiss_gpu'
Faiss falling back to CPU-only.
Reading source word embeddings...
Done.
(200000, 300)
Reading target word embeddings...
Done.
(200000, 300)
Reading validation file...
Reading gold file...
Constructing source word-id map...
Done.
Constructing target word-id map...
Everything Done.
```
Run main.py in mode-1 to start training. In the end it will calculate P@k, and generate a plot of discriminator accuracy, generator loss vs epochs
```
$ python main.py --mode 1
Reading embedding numpy files...
Done.
Converting arrays to embedding layers...
Done.
Epoch 0 : Discriminator Loss: 0.67081, Discriminator Accuracy: 0.63142, Generator Loss: 0.73843, Time elapsed 2.72 mins
Epoch 1 : Discriminator Loss: 0.67737, Discriminator Accuracy: 0.63803, Generator Loss: 0.71354, Time elapsed 2.71 mins
P@5 : 0.0
```
Run main.py in mode-2 for evaluation. The saved model should be kept in models/ directory. P@k gets calculated for all models (adv, procrustes), methods (csls, nn), refinement/no-refinement, 2 validation sets and k = 1, 5, 10. You can selectively run evaluations by modifying _get_eval_params function in main.py. Full evaluation takes ~15 min.
```
$ python main.py --mode 2
Reading embedding numpy files...
Done.
Converting arrays to embedding layers...
Done.
...
```
![alt text](https://github.com/MysteryVaibhav/vanGAN/blob/gan/src/d_g.png)

Architecture choices (https://arxiv.org/abs/1710.04087):
------------------------------
- Discriminator:
    (a) 2 hidden layers, 2048 size each, Leaky-ReLU activation functions
    (b) Input corrupted with a dropout rate of 0.1
    (c) Smoothing coefficient (s = 0.2) in discriminator output (GoodFellow - 2016)
- SGD, batch_size = 32, learning rate = 0.1, decay = 0.95 for both discriminator and generator
- "unsupervised validation criteria" decreases => divide learning rate by 2
- Feed discriminator with only the most frequent 50K words (sampled uniformly)
- Orthogonality of W: Alternate update rule with W ← (1 + β)W − β(W.W')W (β = 0.01)
- Unsupervised validation criteria:
    (a) Consider the 10k most frequent source words
    (b) Use CSLS to generate a translation for each of them
    (c) Compute the average cosine similarity between these deemed translations
    (d) Use this average as a validation metric
    
Initial Results on Fastext embeddings:
------------------------------
Stopped after 4 epochs.
For the best model:
- P@1 : 0.4066666666666667
- P@5 : 0.6153333333333333
- P@10 :  0.684
```
Epoch 0 : Discriminator Loss: 0.41995, Discriminator Accuracy: 0.93694, Generator Loss: 1.82615, Time elapsed 105.70 mins
P@5: 0.5353333333333333
Epoch 1 : Discriminator Loss: 0.37931, Discriminator Accuracy: 0.97755, Generator Loss: 2.00314, Time elapsed 114.67 mins
P@5: 0.6153333333333333
Epoch 2 : Discriminator Loss: 0.36378, Discriminator Accuracy: 0.99002, Generator Loss: 2.04785, Time elapsed 139.61 mins
P@5: 0.468
Epoch 3 : Discriminator Loss: 0.35603, Discriminator Accuracy: 0.99487, Generator Loss: 2.06284, Time elapsed 150.38 mins
P@5: 0.37466666666666665
[9562/31250] :: Discriminator Loss: 0.352862     Generator Loss: 2.067446
```

Installing Faiss (on Linux GPU):
------------------------------
- Delete the faiss package from the repo.
- Clone the Faiss repository inside vanGAN: git clone https://github.com/facebookresearch/faiss.git
- Step-1: C++ Compilation:
    - Open the MakeFile. Set MAKEFILE_INC=example_makefiles/makefile.inc.Linux
    - Go to example_makefiles/makefile.inc.Linux
    - Make the following changes to this file:
        - Comment the BLASLD flag for CentOS (line after #This is for Centos)
        - Install OpenBLAS for Ubuntu 16/14 using the command given in the file. Once installed, comment the line.
        - Uncomment the BLASLD flag for Ubuntu 16/14 (my GPU has Ubuntu 16; verify which one is yours using "lsb_release -a" command
        - Now go back to faiss/ directory and run: make tests/test_blas. This should run without error.
        - Verify everything is working fine by running ./tests/test_blas
- Step-2: Python Interface Compilation:
    - Go to example_makefiles/makefile.inc.Linux
    - Modify the PYTHONCFLAGS to point to your own Python installation. I used the following flag:
        PYTHONCFLAGS=-I/home/ubuntu/anaconda3/include/python3.6m/ -I/home/ubuntu/anaconda3/lib/python3.6/site-packages/numpy/core/include/
    - Go back to faiss/ directory and run: make py. This should run without errors.
    - Test faiss installation by running python -c "import faiss". It will display an error message that GPU not enabled and using the CPU installation. This is fine.
- Step-3: You might have to change module import statements inside validation_faiss.py and faiss.py in case of ModuleNotFound errors.
