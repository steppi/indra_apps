import os
import pickle

from deft.discover import DeftMiner

data_path = os.path.join('/deft_drive', 'indra_apps', 'deft',
                         'input')
inpath = os.path.join(data_path, 'fulltexts')
outpath = os.path.join(data_path, 'longforms')


def mine(filepath):
    filename = os.path.basename(filepath)
    shortform = filename.split('_')[0]
    with open(filepath, 'rb') as f:
        texts = pickle.load(f)
    texts = [text for text in texts if text]
    dm = DeftMiner(shortform)
    dm.process_texts(texts)
    longforms = dm.get_longforms()
    outfile = os.path.join(outpath, f'{shortform}_longforms.pkl')
    with open(outfile, 'wb') as f:
        pickle.dump(longforms, f)


if __name__ == '__main__':
    for fl in os.listdir(inpath):
        filename = os.fsdecode(fl)
        if filename.endswith('.pkl'):
            filepath = os.path.join(inpath, filename)
            mine(filepath)
