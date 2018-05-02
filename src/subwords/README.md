# GAN with subword embeddings

## Preparation

Download [pre-trained models (bin+text)](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) of English, French, Spanish, German, Russian and Chinese to `../../data` folder. Decompress them.

At this directory, run `./setup.sh`. This process takes a while.


| lang | # of subwords within top 10k words |    75k |
|------+------------------------------------+--------|
| fr   |                              72886 | 389406 |
| en   |                              73889 | 411769 |
| es   |                              72776 | 382598 |
| de   |                              75534 | 386907 |
| it   |                              68970 | 362237 |
| zh   |                              55129 | 361377 |
| ru   |                              74380 | 380889 |



# Experiments

```shell
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --model_file ../../models/wiki.en-es/generator_weights_en_es_seed_100_75000_0.2_0.696.t7 --validation_file en-es.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.es.words.npz
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --model_file ../../models/wiki.en-ru/generator_weights_en_ru_seed_988_mf_75000_lr_0.2_p@1_34.330.t7 --validation_file en-ru.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.ru.words.npz
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --model_file ../../models/wiki.en-zh/generator_weights_en_zh_seed_394_mf_50000_lr_0.2_p@1_17.530.t7 --validation_file en-zh.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.zh.words.npz
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --model_file ../../models/wiki.en-de/generator_weights_en_de_seed_310_mf_75000_lr_0.2_p@1_62.470.t7 --validation_file en-de.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.de.words.npz
```
