# dist-based-model4bwe
PyTorch implementation of "A Distribution-based Model to Learn Bilingual Word Embeddings" (Cao et al., COLING2016)


```shell
conda install pytorch -c pytorch -y
conda install gensim
conda install tqdm
```

```shell
python main.py --src en:data/ukWaC/tokenized.1k.txt.xz --trg it:data/itWaC/tokenized.1k.txt.xz -o vectors.txt --batch-size 1024 --iter 100 --model model/en.it/1k/ -v
```

```
2018-02-20 09:39:20,927/Corpus[INFO]: Read from data/ukWaC/tokenized.1k.txt.xz
2018-02-20 09:39:20,963/Corpus[INFO]: Done.
2018-02-20 09:39:20,964/Corpus[INFO]: Read from data/itWaC/tokenized.1k.txt.xz
2018-02-20 09:39:21,006/Corpus[INFO]: Done.
2018-02-20 09:39:21,216/MAIN[INFO]: window size: 2
2018-02-20 09:39:21,216/MAIN[INFO]: learning rate: 0.01
2018-02-20 09:39:21,216/MAIN[INFO]: batch size: 1024
28it [00:03,  7.12it/s]
[1] loss = -767.1751 (-822.5153/55.3402), time = 3.93
2018-02-20 09:39:25,150/MAIN[INFO]: Save embeddings to vectors.txt
28it [00:03,  9.00it/s]
[2] loss = -805.2450 (-822.1792/16.9341), time = 3.11
2018-02-20 09:39:28,674/MAIN[INFO]: Save embeddings to vectors.txt
28it [00:06,  4.65it/s]
[3] loss = -806.3456 (-823.2789/16.9333), time = 6.02
2018-02-20 09:39:35,076/MAIN[INFO]: Save embeddings to vectors.txt
28it [00:03,  8.69it/s]
[4] loss = -807.0366 (-823.9692/16.9327), time = 3.22
2018-02-20 09:39:38,746/MAIN[INFO]: Save embeddings to vectors.txt
28it [00:03,  8.87it/s]
[5] loss = -810.8682 (-827.8003/16.9320), time = 3.16
2018-02-20 09:39:42,306/MAIN[INFO]: Save embeddings to vectors.txt
```

3x faster if you use GPU when `batchsize = 1024`.


```python
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('vectors.txt', binary=False)
model.most_similar('en:apple')
> [('it:alan', 0.5561162829399109), ('it:rifiuto', 0.48470214009284973), ('it:rivendicazione', 0.47415250539779663), ('en:picture', 0.43885284662246704), ('en:children', 0.4348059296607971), ('it:quarto', 0.4281408190727234), ('en:he.&apos;', 0.4273384213447571), ('it:sirena', 0.4250766634941101), ('en:enemy', 0.4240286350250244), ('it:adesso', 0.42393574118614197)]
```

You can get a better result if you use larger corpora.
