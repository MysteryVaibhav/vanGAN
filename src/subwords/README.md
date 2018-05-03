# GAN with subword embeddings

## Preparation

Download [pre-trained models (bin+text)](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) of English, French, Spanish, German, Russian and Chinese to `../../data` folder. Decompress them.

At this directory, run `./setup.sh`. This process takes a while.


| lang | # of subwords within top 10k words |    75k |   200k |
|------+------------------------------------+--------+--------|
| fr   |                              72886 | 389406 |        |
| en   |                              73889 | 411769 | 906017 |
| es   |                              72776 | 382598 |        |
| de   |                              75534 | 386907 |        |
| it   |                              68970 | 362237 |        |
| zh   |                              55129 | 361377 |        |
| ru   |                              74380 | 380889 |        |



# Experiments

```shell
mkdir -p ../../model/wiki.en-zh/subwords/scratch && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --validation_file en-zh.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.zh.words.npz --model_dir ../../model/wiki.en-zh/subwords/scratch
mkdir -p ../../model/wiki.en-es/subwords/scratch && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --validation_file en-es.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.es.words.npz --model_dir ../../model/wiki.en-es/subwords/scratch

# w/ initialization
mkdir -p ../../model/wiki.en-zh/subwords && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --model_file ../../models/wiki.en-zh/generator_weights_en_zh_seed_394_mf_50000_lr_0.2_p@1_17.530.t7 --validation_file en-zh.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.zh.words.npz --model_dir ../../model/wiki.en-zh/subwords
mkdir -p ../../model/wiki.en-es/subwords && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --model_file ../../models/wiki.en-es/generator_weights_en_es_seed_100_75000_0.2_0.696.t7 --validation_file en-es.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.es.words.npz --model_dir ../../model/wiki.en-es/subwords


mkdir -p ../../model/wiki.en-ru/subwords && CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python -m pdb main.py --mode 1 --model_file ../../models/wiki.en-ru/generator_weights_en_ru_seed_988_mf_75000_lr_0.2_p@1_34.330.t7 --model_dir ../../model/wiki.en-ru/subwords



mkdir -p ../../model/wiki.en-de/subwords && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --model_file ../../models/wiki.en-de/generator_weights_en_de_seed_310_mf_75000_lr_0.2_p@1_62.470.t7 --validation_file en-de.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.de.words.npz --model_dir ../../model/wiki.en-zh/subwords


# w/ Distribution-based Initialization
mkdir -p ../../model/wiki.en-es/subwords/dist_init && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --validation_file en-es.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.es.words.npz --model_dir ../../model/wiki.en-es/subwords/dist_init
mkdir -p ../../model/wiki.enzh/subwords/dist_init && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main.py --mode 1 --validation_file enzh.5000-6500.subwords --src_file wiki.en.subwords.top75000.npz --tgt_file wiki.zh.words.npz --model_dir ../../model/wiki.enzh/subwords/dist_init
```


## Output Word Embeddings Using Resulting Subword Transformer

```shell
PYTHONPATH=.:$PYTHONPATH python scripts/output_word_embeddings.py --subwords ../../data/wiki.en.subwords.topn200000.npz -o ../../model/wiki.en-zh/subwords/wiki.en.e10.vec --transformer ../../model/wiki.en-zh/subwords/s_e9.pth --original ../../data/wiki.en.vec -v
```
