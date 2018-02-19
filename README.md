# vanGAN
11-747 Course Project: Bilingual Lexicon Induction

Steps for compiling:
- Edit the properties file to set the path of the data directory (Download data from http://clic.cimec.unitn.it/~georgiana.dinu/down/)
- Run the util file once, it reads the datafiles and creates a numpy array for faster access
- Run the trainer to start training

Things to do:
- Add initilization and update constraints for the parameters (orthogonality)
- Can add gradient cilpping if training is erratic
- Add noise while training the generator and discriminator
- Change the simple binary cross entropy loss to something more complicated
- Add method to retrieve the nearest neighbours from other language (trivial if using KNN)
- Remove properties file and add everything to program arguments (can be deferred for now)
