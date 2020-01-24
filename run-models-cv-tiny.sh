#!/bin/bash

NUMBER_OF_FOLDS=${1:-10}
NER_COMPARISON_HOME=/home/nami/ner-comparison

. ./python/bin/activate
LD_LIBRARY_PATH=/usr/local/cuda/lib64

model=ner_rus_bert
i=01
mkdir $NER_COMPARISON_HOME/cv/$model
python train-and-predict.py --tagged --model $model --input $NER_COMPARISON_HOME/cv/reference/$i.txt --output $NER_COMPARISON_HOME/cv/$model/$i.txt --train ~/fact-ru-eval2stanford/conll2003rucv/$i/train.txt --test ~/fact-ru-eval2stanford/conll2003rucv/$i/test.txt --valid ~/fact-ru-eval2stanford/conll2003rucv/$i/test.txt

deactivate
