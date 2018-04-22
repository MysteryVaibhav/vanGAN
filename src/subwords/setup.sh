DIR_SCRIPT=../../scripts
DIR_DATA=../../data

for lang in fr en es de it ru zh
do
    if [ ! -e ${DIR_DATA}/wiki.${lang}.subwords ]; then
        cmd="python ${DIR_SCRIPT}/decompose_words.py ${DIR_DATA}/wiki.${lang} -v"
        echo $cmd
        eval $cmd
    fi
    if [ ! -e ${DIR_DATA}/wiki.${lang}.subwords.npz ]; then
        cmd="python ${DIR_SCRIPT}/extract_subword_embeddings.py ${DIR_DATA}/wiki.${lang} -v"
        echo $cmd
        eval $cmd
    fi
done
