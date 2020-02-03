#!/bin/bash

NUMBER_OF_FOLDS=${1:-10}

for model in ner_rus #ner_bert_slav #ner_ontonotes_bert_mult #ner_ontonotes_bert #ner_rus_bert
do
	mkdir -p custom-configs/$model
	for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
	do
		cp custom-configs/$model.json custom-configs/$model/$model-$i.json
		sed -i -E "s|cv/01/|cv/$i/|" custom-configs/$model/$model-$i.json
	done
done
