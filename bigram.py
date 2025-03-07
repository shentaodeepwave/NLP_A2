import argparse
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import math
from tqdm import tqdm
nltk.download('punkt')

def preprocess(inputfile, outputfile):
    with open(inputfile, 'r', encoding='utf-8') as infile, open(outputfile, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.lower()
            tokens = word_tokenize(line)
            tokens = ['<s>'] + tokens + ['</s>']
            outfile.write(' '.join(tokens) + '\n')

def sentence_preprocess(sentence, word_dict):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    tokens = ['<s>'] + [token if token in word_dict else '<UNK>' for token in tokens] + ['</s>']
    return tokens

def count_word(inputfile, outputfile):
    word_count = defaultdict(int)
    with open(inputfile, 'r', encoding='utf-8') as infile:
        for line in infile:
            tokens = line.split()
            for token in tokens:
                word_count[token] += 1
    with open(outputfile, 'w', encoding='utf-8') as outfile:
        for word, count in word_count.items():
            outfile.write(f'{word} {count}\n')

def count_bigram(inputfile, outputfile):
    bigram_count = defaultdict(int)
    with open(inputfile, 'r', encoding='utf-8') as infile:
        for line in infile:
            tokens = line.split()
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                bigram_count[bigram] += 1
    with open(outputfile, 'w', encoding='utf-8') as outfile:
        for bigram, count in bigram_count.items():
            outfile.write(f'{bigram[0]} {bigram[1]} {count}\n')

def read_word_count():
    word_dict = {}
    with open('word.txt', 'r', encoding='utf-8') as infile:
        for line in infile:
            word, count = line.split()
            word_dict[word] = int(count)
    return word_dict

def read_bigram_count():
    bigram_dict = {}
    with open('bigram.txt', 'r', encoding='utf-8') as infile:
        for line in infile:
            word1, word2, count = line.split()
            bigram_dict[(word1, word2)] = int(count)
    return bigram_dict

def add_one_perplexity(sentence):
    word_dict = read_word_count()
    bigram_dict = read_bigram_count()
    sentence = sentence_preprocess(sentence, word_dict)
    V = len(word_dict)
    N = len(sentence) - 1
    perplexity = 1.0
    for i in range(N):
        bigram = (sentence[i], sentence[i + 1])
        bigram_count = bigram_dict.get(bigram, 0) + 1
        word_count = word_dict.get(sentence[i], 0) + V
        perplexity *= (bigram_count / word_count)
    perplexity = math.pow(perplexity, -1 / N)
    return perplexity

def add_n_perplexity(sentence, n):
    word_dict = read_word_count()
    bigram_dict = read_bigram_count()
    sentence = sentence_preprocess(sentence, word_dict)
    V = len(word_dict)
    N = len(sentence) - 1
    perplexity = 1.0
    for i in range(N):
        bigram = (sentence[i], sentence[i + 1])
        bigram_count = bigram_dict.get(bigram, 0) + n
        word_count = word_dict.get(sentence[i], 0) + n * V
        perplexity *= (bigram_count / word_count)
    perplexity = math.pow(perplexity, -1 / N)
    return perplexity

def add_n_perplexity_batch(input, output, n):
    word_dict = read_word_count()
    bigram_dict = read_bigram_count()
    total_perplexity = 0.0
    sentence_count = 0

    with open(input, 'r', encoding='utf-8') as infile, open(output, 'w', encoding='utf-8') as outfile:
        outfile.write(f'test-Set-PPL\n')
        for line in tqdm(infile):
            sentence_original = line.strip()
            sentence = sentence_preprocess(sentence_original, word_dict)
            
            V = len(word_dict)
            N = len(sentence)
            perplexity = 1.0
            for i in range(N-1):
                bigram = (sentence[i], sentence[i + 1])
                bigram_count = bigram_dict.get(bigram, 0) + n
                word_count = word_dict.get(sentence[i], 0) + n * V
                perplexity *= (bigram_count / word_count)
            if perplexity != 0:

                perplexity = math.pow(perplexity, -1 / (N))
                total_perplexity += perplexity
                sentence_count += 1
            else:
                perplexity = str('INF')

            outfile.write(f'{sentence_original} {perplexity}\n')
        average_perplexity = total_perplexity / sentence_count
        print('Average Perplexity: ' + str(average_perplexity))

def main():
    ''' Main Function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-pps', '--preprocess', type=str, nargs=2, help='preprocess the dataset')
    parser.add_argument('-cw', '--count_word', type=str, nargs=2, help='count the words from the corpus')
    parser.add_argument('-cb', '--count_bigram', type=str, nargs=2, help='count the bigrams from the corpus')
    parser.add_argument('-ppl1', '--add_one_perplexity', type=str, nargs=1,
                        help='calculate the perplexity of the sentence using the add-1 smoothing')
    parser.add_argument('-ppln', '--add_n_perplexity', type=str, nargs=2,
                        help='calculate the perplexity of the sentence using the add-n smoothing')
    parser.add_argument('-pplnb', '--add_n_perplexity_batch', type=str, nargs=3,
                        help='calculate the perplexity of the sentence using the add-n smoothing')
    opt = parser.parse_args()

    if opt.preprocess:
        input_file = opt.preprocess[0]
        output_file = opt.preprocess[1]
        print('Preprocessing the dataset...')
        preprocess(input_file, output_file)
    elif opt.count_word:
        input_file = opt.count_word[0]
        output_file = opt.count_word[1]
        count_word(input_file, output_file)
    elif opt.count_bigram:
        input_file = opt.count_bigram[0]
        output_file = opt.count_bigram[1]
        count_bigram(input_file, output_file)
    elif opt.add_one_perplexity:
        ppl = add_one_perplexity(opt.add_one_perplexity[0])
        print('The perplexity of the sentence using the add-one smoothing is: ' + str(ppl))
    elif opt.add_n_perplexity:
        sentence = opt.add_n_perplexity[0]
        n = int(opt.add_n_perplexity[1])
        ppl = add_n_perplexity(sentence, n)
        print('The perplexity of the sentence using the add-' + str(n) + ' smoothing is: ' + str(ppl))
    elif opt.add_n_perplexity_batch:
        input = opt.add_n_perplexity_batch[0]
        output = opt.add_n_perplexity_batch[1]
        n = int(opt.add_n_perplexity_batch[2])
        add_n_perplexity_batch(input, output, n)

if __name__ == '__main__':
    import os
    main()