#!/bin/bash

NUMBER_OF_FOLDS=${1:-10}
NER_COMPARISON_HOME=/home/nami/ner-comparison

. ./python/bin/activate
LD_LIBRARY_PATH=/usr/local/cuda/lib64

for model in ner_rus
do
	mkdir -p $NER_COMPARISON_HOME/cv-ner-rus-with-fine-tuning/$model
	for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
	do
		rm -rf ~/.deeppavlov/models/$model
		python -m deeppavlov train custom-configs/$model/$model-$i.json
		python predict.py --tagged --model $model --input $NER_COMPARISON_HOME/cv/reference/$i.txt --output $NER_COMPARISON_HOME/cv-ner-rus-with-fine-tuning/$model/$i.txt
	done
done

deactivate

#python -m deeppavlov train custom-configs/ner_rus_bert/ner_rus_bert-01.json
