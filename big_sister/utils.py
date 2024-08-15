import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

# splits the prompt into an array of words
def tokenize(sentence):
  return nltk.word_tokenize(sentence)

# lowercases the words and removes endings
def stem(word):
  return stemmer.stem(word.lower())

# takes a tokenized sentence array and all_words array:
# 1. stems it
# 2. makes the array all zeros
# 3. checks if each word in tokenized_sentence is in all_words
# 4. if yes moves changes it to 1
def bag_of_words(tokenized_sentence, all_words):
  sentence = [stem(word) for word in tokenized_sentence]

  bag = np.zeros(len(all_words), dtype=np.float32) # makes all words 0
  for ind, word in enumerate(all_words):
    if word in sentence:
      bag[ind] = 1

  return bag

