import os
import pickle
from nltk.tokenize import PunktSentenceTokenizer


def get_articles(path):
    output = []
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith('.txt'):
                with open(filepath, 'r') as f:
                    output.append(f.read())
    return output


_root_path = '../input/pmc_articles/'
_journals = ['BMC_Bioinformatics',
             'BMC_Biochem',
             'BMC_Cancer',
             'BMC_Cell_Biol',
             'BMC_Biol',
             'BMC_Mol_Biol',
             'Biochemistry',
             'Bioinformatics',
             'Biol_Cell',
             'Cancer',
             'Cell']
             

if __name__ == '__main__':
    training_corpus = []
    for journal in _journals:
        training_corpus.extend(get_articles(_root_path +
                                            journal))

    training_corpus = " ".join(training_corpus)
    tokenizer = PunktSentenceTokenizer()
    tokenizer.train(training_corpus)
    with open('bio_sent_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
