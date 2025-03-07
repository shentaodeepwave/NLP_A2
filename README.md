# Bigram Language Model

This is a program that uses a Bigram Model to calculate the perplexity of sentences. The program includes data preprocessing, word frequency statistics, bigram statistics, and the calculation of perplexity using additive smoothing.

## Dependencies

Before running this program, please ensure that the following dependencies are installed:

- Python 3.x
- NLTK

You can install the dependencies using the following command:

```bash
pip install nltk
```

## Usage

The program executes different functions through command-line arguments. Below are the usage methods for each function:

### Data Preprocessing

Preprocess the text in the input file, including converting the text to lowercase, tokenizing, and adding special markers `<s>` and `</s>` at the beginning and end of each line.

```bash
python bigram.py -pps "news.train" "corpus.txt" 
```

### Word Frequency Statistics

Count the word frequency in the input file and save the results to the output file.

```bash
python bigram.py -cw "corpus.txt" "word.txt"
```

### Bigram Statistics

Count the bigram frequency in the input file and save the results to the output file.

```bash
python bigram.py -cb "corpus.txt" "bigram.txt"
```

### Calculate Add-One Smoothing Perplexity of a Sentence

Calculate the add-one smoothing perplexity of a given sentence.

```bash
python bigram.py -ppl1 "This is a test sentence."  
```

### Calculate Add-N Smoothing Perplexity of a Sentence

Calculate the add-n smoothing perplexity of a given sentence.

```bash
python bigram.py -ppln "This is a test sentence." 5 
```

### Batch Calculate Add-N Smoothing Perplexity of Sentences

Batch calculate the add-n smoothing perplexity of each sentence in the input file and save the results to the output file.

```bash
python bigram.py -pplnb "news.test" "perplexity-n.txt" 0.00425
```