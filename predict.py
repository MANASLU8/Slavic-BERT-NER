import argparse, os
from deeppavlov import build_model, configs
from converters import tokenize
from nltk.tokenize import RegexpTokenizer
from file_operations import write_lines

def write_prediction_results(labels, tokens, output_file):
    lines = []
    for sentence_tokens, sentence_labels in zip(tokens, labels):
        for token, label in zip(sentence_tokens, sentence_labels):
            lines.append(f'{token} {label.split("-")[-1].capitalize()}')
    write_lines(output_file, lines)

def extract_predictions(sentence):
    return list(map(lambda one_item_list: one_item_list[0], sentence[1]))

if __name__ == "__main__":
    #print(configs.ner.ner_rus_bert)
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default='raw.txt')
    parser.add_argument('--output_file', type=str, default='raw.predictions.txt')

    args = parser.parse_args()

    tokens = tokenize(args.input_file, tokenizer)

    # Download and load model (set download=False to skip download phase)
    ner = build_model("./ner_bert_slav.json", download=True)
    #ner = build_model(configs.ner.ner_rus_bert, download=True)

    entities = list(map(lambda sentence: extract_predictions(ner(sentence)), tokens))

    write_prediction_results(entities, tokens, args.output_file)

