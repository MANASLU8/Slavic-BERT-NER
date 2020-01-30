import argparse, os
from deeppavlov import build_model, configs
from deeppavlov import train_evaluate_model_from_config
from converters import tokenize, tokenize_tagged, NO_ENTITY_MARK, lines_to_x_y_pairs
from nltk.tokenize import RegexpTokenizer
from file_operations import write_lines, read_lines, read
import tensorflow as tf
import json
from deeppavlov.core.trainers import NNTrainer
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from keras.backend.tensorflow_backend import set_session
from deeppavlov.core.commands.utils import import_packages, parse_config

from deeppavlov.download import deep_download
from deeppavlov.core.common.chainer import Chainer

config = tf.ConfigProto()

config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU

config.log_device_placement = True

sess = tf.Session(config=config)

set_session(sess)

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


def make_dataset_iterator_from_conll2003(train_file, test_file, valid_file, config_id):
    train_pairs = lines_to_x_y_pairs(read_lines(train_file), config_id)
    test_pairs = lines_to_x_y_pairs(read_lines(test_file), config_id)
    valid_pairs = lines_to_x_y_pairs(read_lines(valid_file), config_id)
    print(train_pairs[:100])
    print(test_pairs[:100])
    print(valid_pairs[:100])
    return DataLearningIterator({'train': train_pairs, 'test': test_pairs, 'valid': valid_pairs}, shuffle=False)


def train(config_id, train_file, test_file, valid_file):
    config = parse_config(MODEL_CONFIGS[config_id])
    print(MODEL_CONFIGS[config_id])
    deep_download(config)

    import_packages(config.get('metadata', {}).get('imports', []))

    model_config = config['chainer']

    model = Chainer(model_config['in'], model_config['out'], model_config.get('in_y'))

    ner = NNTrainer(model_config,
         batch_size = config['train']['epochs'],
         epochs =  config['train']['batch_size'],
         start_epoch_num = 0,
         metrics = config['train']['metrics'],
         train_metrics =  config['train']['metrics'],
         metric_optimization = 'maximize',
         evaluation_targets = config['train']['evaluation_targets'],
         validation_patience = config['train']['validation_patience'],
         val_every_n_batches = config['train']['val_every_n_epochs'],
         #log_every_n_batches = config['train']['log_every_n_batches'],
         show_examples = config['train']['show_examples'])
    ner._chainer = build_model(MODEL_CONFIGS[config_id], download = True)

    dataset_iterator = make_dataset_iterator_from_conll2003(train_file, test_file, valid_file, config_id)
    ner.train(dataset_iterator)

    return ner._chainer

    # use default train command

    # print(MODEL_CONFIGS[config_id])
    # dataset_iterator = make_dataset_iterator_from_conll2003(train_file, test_file, valid_file, config_id)
    # #ner.train(dataset_iterator)
    # return train_evaluate_model_from_config(MODEL_CONFIGS[config_id], dataset_iterator, download = True)

def _predict(model, tokens, output_file):
    # Download and load model (set download=False to skip download phase)
    #print(f'Config: {MODEL_CONFIGS[config_id]}')
    #ner = build_model(MODEL_CONFIGS[config_id], download = True)

    # if args.model == MODEL_IDS['slavic']:
    #     ner = build_model("./ner_bert_slav.json", download=True)
    # elif args.model == MODEL_IDS['rus']:
    #     ner = build_model(configs.ner.ner_rus_bert, download=True)

    #print(dir(ner))

    entities = list(map(lambda sentence: extract_predictions(model(sentence)), tokens))

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
        print('training...')
        model = train(get_key(MODEL_IDS, args.model), args.train, args.test, args.valid)
        print('predicting...')
        _predict(model, tokens, args.output_file)
    else:
        for config in MODEL_CONFIGS:
            #print(config)
            _predict(config, tokens, f'eval.tagged.{config}.txt')
