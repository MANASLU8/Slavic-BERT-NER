import argparse, os
from deeppavlov import build_model, configs
from converters import tokenize, tokenize_tagged, NO_ENTITY_MARK
from nltk.tokenize import RegexpTokenizer
from file_operations import write_lines

TAGGED_MARK = 'tagged'

PREDICTED_TAG_BODIES_MAPPING = {'LOC': 'Location', 'ORG': 'Org', 'PER': 'Person', 'O': 'O'}
DEFAULT_TAG_BODY = 'Other'

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

if __name__ == "__main__":
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default='raw.txt')
    parser.add_argument('--output_file', type=str, default='raw.predictions.txt')

    args = parser.parse_args()

    tokens = tokenize(args.input_file, tokenizer) if not TAGGED_MARK in args.input_file.split('/')[-1].split('.') else tokenize_tagged(args.input_file)

    # Download and load model (set download=False to skip download phase)
    ner = build_model("./ner_bert_slav.json", download=True)
    #ner = build_model(configs.ner.ner_rus_bert, download=True)

    entities = list(map(lambda sentence: extract_predictions(ner(sentence)), tokens))

    write_prediction_results(entities, tokens, args.output_file)
