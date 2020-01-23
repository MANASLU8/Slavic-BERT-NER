#!/bin/bash

NUMBER_OF_FOLDS=${1:-10}
NER_COMPARISON_HOME=/home/nami/ner-comparison

. ./python/bin/activate
LD_LIBRARY_PATH=/usr/local/cuda/lib64

for model in ner_ontonotes_bert ner_ontonotes_bert_mult ner_rus_bert ner_rus ner_ontonotes ner_ontonotes slavic
do
	mkdir $NER_COMPARISON_HOME/cv/$model
	for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
	do
		python predict.py --tagged --model $model --input $NER_COMPARISON_HOME/cv/reference/$i.txt --output $NER_COMPARISON_HOME/cv/$model/$i.txt
	done
done

deactivate