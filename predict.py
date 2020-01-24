import argparse, os
from deeppavlov import build_model, configs
from converters import tokenize, tokenize_tagged, NO_ENTITY_MARK, lines_to_x_y_pairs
from nltk.tokenize import RegexpTokenizer
from file_operations import write_lines, read_lines, read

import json
from deeppavlov.core.trainers import NNTrainer
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

from deeppavlov.core.commands.utils import import_packages, parse_config

from deeppavlov.download import deep_download
from deeppavlov.core.common.chainer import Chainer

TAGGED_MARK = 'tagged'

PREDICTED_TAG_BODIES_MAPPING = {'LOC': 'Location', 'ORG': 'Org', 'PER': 'Person', 'O': 'O', 'PERSON': 'Person', 
    'GPE': 'Org', 'NORP': 'Location', 'WORK_OF_ART': 'Org'}
DEFAULT_TAG_BODY = 'Other'

MODEL_IDS = {
    'slavic': 'slavic',
    'ontonotes_bert': 'ner_ontonotes_bert',
    'ontonotes_mult': 'ner_ontonotes_bert_mult',
    'rus_bert': 'ner_rus_bert',
    'rus': 'ner_rus',
    'few_shot_simulate': 'ner_few_shot_ru_simulate',
    'ontonotes': 'ner_ontonotes',
    'few_shot': 'ner_few_shot_ru',
    'kb_rus': 'ner_kb_rus'
}

MODEL_CONFIGS = {
    'slavic': './ner_bert_slav.json',
    'ontonotes_bert': configs.ner.ner_ontonotes_bert,
    'ontonotes_mult': configs.ner.ner_ontonotes_bert_mult,
    'rus_bert': configs.ner.ner_rus_bert,
    'rus': configs.ner.ner_rus,
    'few_shot_simulate': configs.ner.ner_few_shot_ru_simulate,
    'ontonotes': configs.ner.ner_ontonotes,
    'few_shot': configs.ner.ner_few_shot_ru,
    'kb_rus': configs.ner.ner_kb_rus
}

def get_key(dictionary, required_value):
    for key, value in dictionary.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if value == required_value:
            return key

def write_prediction_results(labels, tokens, output_file):
    lines = []
    for sentence_tokens, sentence_labels in zip(tokens, labels):
        for token, label in zip(sentence_tokens, sentence_labels):
            if len(label.split("-")) < 2:
                lines.append(f'{token} {NO_ENTITY_MARK}')
            else:
                tag_prefix, tag_body = label.split("-")
                tag_body = PREDICTED_TAG_BODIES_MAPPING.get(tag_body, DEFAULT_TAG_BODY)
                lines.append(f'{token} {tag_prefix}-{tag_body}')
            #lines.append(f'{token} {label}')
            #lines.append(f'{token} {label.split("-")[-1].capitalize()}')
        lines.append('')
    write_lines(output_file, lines)

def extract_predictions(sentence):
    return list(map(lambda one_item_list: one_item_list[0], sentence[1]))


def make_dataset_iterator_from_conll2003(train_file, test_file, valid_file):
    train_pairs = lines_to_x_y_pairs(read_lines(train_file))
    test_pairs = lines_to_x_y_pairs(read_lines(test_file))
    valid_file = lines_to_x_y_pairs(read_lines(valid_file))
    return DataLearningIterator({'train': train_pairs, 'test': test_pairs, 'valid': valid_file})


def train(config_id, train_file, test_file, valid_file):
    config = parse_config(MODEL_CONFIGS[config_id])
    deep_download(config)

    import_packages(config.get('metadata', {}).get('imports', []))

    model_config = config['chainer']

    model = Chainer(model_config['in'], model_config['out'], model_config.get('in_y'))

    ner = NNTrainer(model_config)
    ner._chainer = model

    dataset_iterator = make_dataset_iterator_from_conll2003(train_file, test_file, valid_file)
    ner.train(dataset_iterator)
    return ner

def _predict(config_id, tokens, output_file):
    # Download and load model (set download=False to skip download phase)
    print(f'Config: {MODEL_CONFIGS[config_id]}')
    #ner = build_model(MODEL_CONFIGS[config_id], download = True)

    # if args.model == MODEL_IDS['slavic']:
    #     ner = build_model("./ner_bert_slav.json", download=True)
    # elif args.model == MODEL_IDS['rus']:
    #     ner = build_model(configs.ner.ner_rus_bert, download=True)

    #print(dir(ner))

    with open(MODEL_CONFIGS[config_id]) as file:
        ner = NNTrainer(json.load(file)["chainer"])

    dd

    entities = list(map(lambda sentence: extract_predictions(ner(sentence)), tokens))

    write_prediction_results(entities, tokens, output_file)

if __name__ == "__main__":
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default='raw.txt')
    parser.add_argument('--output_file', type=str, default='raw.predictions.txt')
    parser.add_argument('--model', type=str, default=MODEL_IDS['slavic'], choices=list(MODEL_IDS.values()))
    parser.add_argument('--run_all', type=bool, default=False)
    parser.add_argument('--tagged', action='store_true')

    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--valid', type=str)

    args = parser.parse_args()

    tokens = tokenize(args.input_file, tokenizer) if not args.tagged and not TAGGED_MARK in args.input_file.split('/')[-1].split('.') else tokenize_tagged(args.input_file)

    if not args.run_all:
        # # Download and load model (set download=False to skip download phase)
        # ner = build_model(MODEL_CONFIGS[get_key(MODEL_IDS, args.model)])
        # # if args.model == MODEL_IDS['slavic']:
        # #     ner = build_model("./ner_bert_slav.json", download=True)
        # # elif args.model == MODEL_IDS['rus']:
        # #     ner = build_model(configs.ner.ner_rus_bert, download=True)

        # entities = list(map(lambda sentence: extract_predictions(ner(sentence)), tokens))

        # write_prediction_results(entities, tokens, args.output_file)
        train(get_key(MODEL_IDS, args.model), args.train, args.test, args.valid)
        print('predicting...')
        _predict(get_key(MODEL_IDS, args.model), tokens, args.output_file)
    else:
        for config in MODEL_CONFIGS:
            #print(config)
            _predict(config, tokens, f'eval.tagged.{config}.txt')
