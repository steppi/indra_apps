import gensim
from gensim.models import KeyedVectors as kv

model = kv.load_word2vec_format('../input/PubMed-and-PMC-w2v.bin',
                                binary=True)

example = model.most_similar(positive=['SHP2', 'RAS', 'SOCS', 'STAM', 'PI3K'],
                             topn=100)
