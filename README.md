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

