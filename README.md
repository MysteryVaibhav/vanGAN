# vanGAN
11-747 Course Project: Bilingual Lexicon Induction

Steps for compiling:
- Edit the properties file to set the path of the data directory (Download data from http://clic.cimec.unitn.it/~georgiana.dinu/down/)
- Run the util file once, it reads the datafiles and creates a numpy array for faster access
- Run the trainer to start training
- (For validation) (a) Uncomment segment of util.py code and run it once. (b) Run validation_faiss.py file.

Things to do:
- Add initilization and update constraints for the parameters (orthogonality)
- Can add gradient cilpping if training is erratic
- Add noise while training the generator and discriminator
- Change the simple binary cross entropy loss to something more complicated
- Add method to retrieve the nearest neighbours from other language (trivial if using KNN)
- Remove properties file and add everything to program arguments (can be deferred for now)

Architecture choices (Facebook paper):
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

Installing Faiss (on Linux GPU):
- Clone the Faiss repository inside vanGAN: git clone https://github.com/facebookresearch/faiss.git
- Step-1: C++ Compilation:
    (a) Open the MakeFile. Set MAKEFILE_INC=example_makefiles/makefile.inc.Linux
    (b) Go to example_makefiles/makefile.inc.Linux
    (c) Make the following changes to this file:
        (i) Comment the BLASLD flag for CentOS (line after #This is for Centos)
        (ii) Install OpenBLAS for Ubuntu 16/14 using the command given in the file. Once installed, comment the line.
        (iii) Uncomment the BLASLD flag for Ubuntu 16/14 (my GPU has Ubuntu 16; verify which one is yours using "lsb_release -a" command
        (iv) Now go back to faiss/ directory and run: make tests/test_blas. This should run without error.
        (v) Verify everything is working fine by running ./tests/test_blas
- Step-2: Python Interface Compilation:
    (a) Go to example_makefiles/makefile.inc.Linux
    (b) Modify the PYTHONCFLAGS to point to your own Python installation. I used the following flag:
        PYTHONCFLAGS=-I/home/ubuntu/anaconda3/include/python3.6m/ -I/home/ubuntu/anaconda3/lib/python3.6/site-packages/numpy/core/include/
    (c) Go back to faiss/ directory and run: make py. This should run without errors.
    (d) Test faiss installation by running python -c "import faiss".
        It will display an error message that GPU not enabled and using the CPU installation. This is fine.

