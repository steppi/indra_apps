import os
import pickle
from nltk.tokenize import PunktSentenceTokenizer


training_corpus = []
for subdir, dirs, files in os.walk('../input/pmc_articles/'):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith('.txt'):
            with open(filepath, 'r') as f:
                training_corpus.append(f.read())

training_corpus = " ".join(training_corpus)
tokenizer = PunktSentenceTokenizer()
tokenizer.train(training_corpus)
with open('bio_sent_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
