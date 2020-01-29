from file_operations import write_lines, read, read_lines

NO_ENTITY_MARK = 'O'

LABEL_MAPPINGS = {
    # 'slavic': './ner_bert_slav.json',
    # 'ontonotes_bert': configs.ner.ner_ontonotes_bert,
    # 'ontonotes_mult': configs.ner.ner_ontonotes_bert_mult,
    'rus': {'Location': 'LOC', 'Person': 'PER', 'LocOrg': 'LOC', 'Org': 'ORG'}
    # 'rus': configs.ner.ner_rus,
    # 'few_shot_simulate': configs.ner.ner_few_shot_ru_simulate,
    # 'ontonotes': configs.ner.ner_ontonotes,
    # 'few_shot': configs.ner.ner_few_shot_ru,
    # 'kb_rus': configs.ner.ner_kb_rus
}

def tokenize(input_file, tokenizer):
    return [sentence for sentence in map(lambda sentence: tokenizer.tokenize(sentence.strip()) + ['.'], read(input_file).split('.')) if len(sentence) > 0]

def tokenize_tagged(input_file):
    lines = read_lines(input_file)
    sentence_tokens = []
    sentences = []
    for line in map(lambda line: line.split(' '), lines):
        if len(line) < 2:
            sentences.append(sentence_tokens)
            sentence_tokens = []
        else:
            sentence_tokens.append(line[0])
    if len(sentence_tokens) > 0:
        sentences.append(sentence_tokens)
    return sentences

def translate_tag_body(tag, mapping):
    splitted_tag = tag.split('-')
    if len(splitted_tag) < 2:
        return splitted_tag
    else:
        return f'{splitted_tag[0]}_{mapping.get(splitted_tag[1], splitted_tag[1])}'


def lines_to_x_y_pairs(lines, config_id):
    pairs = []
    for line in list(map(lambda line: line.split(' '), lines))[2:]:
        if len(line) >= 2:
            pairs.append((line[0],  translate_tag_body(line[-1], LABEL_MAPPINGS[config_id])))
    return pairs