from file_operations import write_lines, read

def tokenize(input_file, tokenizer):
    return [sentence for sentence in map(lambda sentence: tokenizer.tokenize(sentence.strip()) + ['.'], read(input_file).split('.')) if len(sentence) > 0]