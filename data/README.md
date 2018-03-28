# Setting up English-Chinese Evaluation Dataset of MUSE (Conneau et al., 2018)

We found the MUSE evaluation dataset on English-Chinese is written in traditional Chinese characteres. The following command download and convert them into simplified characters.

Prerequisite: [BYVoid/OpenCC: A project for conversion between Traditional and Simplified Chinese](https://github.com/BYVoid/OpenCC)

```shell
wget https://s3.amazonaws.com/arrival/dictionaries/zh-en.txt
wget https://s3.amazonaws.com/arrival/dictionaries/zh-en.0-5000.txt
wget https://s3.amazonaws.com/arrival/dictionaries/zh-en.5000-6500.txt

wget https://s3.amazonaws.com/arrival/dictionaries/en-zh.txt
wget https://s3.amazonaws.com/arrival/dictionaries/en-zh.0-5000.txt
wget https://s3.amazonaws.com/arrival/dictionaries/en-zh.5000-6500.txt

paste -d ' ' <(cut -d ' ' -f 1 zh-en.txt | opencc -c zht2zhs.ini) <(cut -d ' ' -f 2 zh-en.txt) > zhs-en.txt
paste -d ' ' <(cut -d ' ' -f 1 zh-en.0-5000.txt | opencc -c zht2zhs.ini) <(cut -d ' ' -f 2 zh-en.0-5000.txt) > zhs-en.0-5000.txt
paste -d ' ' <(cut -d ' ' -f 1 zh-en.5000-6500.txt | opencc -c zht2zhs.ini) <(cut -d ' ' -f 2 zh-en.5000-6500.txt) > zhs-en.5000-6500.txt

paste -d ' ' <(cut -d ' ' -f 1 en-zh.txt) <(cut -d ' ' -f 2 en-zh.txt | opencc -c zht2zhs.ini) > en-zhs.txt
paste -d ' ' <(cut -d ' ' -f 1 en-zh.0-5000.txt) <(cut -d ' ' -f 2 en-zh.0-5000.txt | opencc -c zht2zhs.ini) > en-zhs.0-5000.txt
paste -d ' ' <(cut -d ' ' -f 1 en-zh.5000-6500.txt) <(cut -d ' ' -f 2 en-zh.5000-6500.txt | opencc -c zht2zhs.ini) > en-zhs.5000-6500.txt
```


# Setting up English-Spanish Data used by Zhang et al. (ACL2017)

__Zh-En__

Test: `zhang2017acl.zh-en.test.txt`

Frequencies and vectors:
```shell
wget https://github.com/muyeby/Bilingual-lexicon-survey/raw/master/UBiLexAT/data/zh-en/vocab-freq.en --output-document zhang2017acl.zh-en.en.freq
wget https://github.com/muyeby/Bilingual-lexicon-survey/raw/master/UBiLexAT/data/zh-en/vocab-freq.zh --output-document zhang2017acl.zh-en.zh.freq
wget https://github.com/muyeby/Bilingual-lexicon-survey/raw/master/UBiLexAT/data/zh-en/word2vec.en --output-document zhang2017acl.zh-en.en.vec
wget https://github.com/muyeby/Bilingual-lexicon-survey/raw/master/UBiLexAT/data/zh-en/word2vec.zh --output-document zhang2017acl.zh-en.zh.vec
```


__Es-En__

Test: `zhang2017acl.es-en.txt`

Frequencies and vectors:
```shell
wget https://github.com/muyeby/Bilingual-lexicon-survey/raw/master/UBiLexAT/data/es-en/vocab-freq.en --output-document zhang2017acl.es-en.en.freq
wget https://github.com/muyeby/Bilingual-lexicon-survey/raw/master/UBiLexAT/data/es-en/vocab-freq.es --output-document zhang2017acl.es-en.es.freq
wget https://github.com/muyeby/Bilingual-lexicon-survey/raw/master/UBiLexAT/data/es-en/word2vec.en --output-document zhang2017acl.es-en.en.vec
wget https://github.com/muyeby/Bilingual-lexicon-survey/raw/master/UBiLexAT/data/es-en/word2vec.es --output-document zhang2017acl.es-en.es.vec
```
