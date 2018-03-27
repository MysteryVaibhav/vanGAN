# Implementation of (Zhang et al., ACL2017)


## Setting up word vectors

```shell
python main.py --data_dir ../../../data/ --src_file zhang2017acl.es-en.es.vec --src_freq_file zhang2017acl.es-en.es.freq --tgt_file zhang2017acl.es-en.en.vec --tgt_freq_file zhang2017acl.es-en.en.freq --validation_file zhang2017acl.es-en.test.txt --mode 0
```


## Run

```shell
python main.py --data_dir ../../../data/  --mode 1
```

A model file will be written to `../../../models/g3.best.pth`
