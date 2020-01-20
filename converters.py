from file_operations import write_lines, read, read_lines

NO_ENTITY_MARK = 'O'

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