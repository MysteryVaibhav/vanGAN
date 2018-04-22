# GAN with subword embeddings

## Preparation

Download [pre-trained models (bin+text)](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) of English, French, Spanish, German, Russian and Chinese to `../../data` folder. Decompress them.

At this directory, run `./setup.sh`. This process takes a while.


lang|# of subwords within top 10k words
----|----------------------------------
fr  |72886
es  |72776
en  |73889
de  |75534
it  |68970
zh  |55129
ru  |74380
