import os
import pickle

from indra.literature.deft_tools import get_plaintexts

path = '/deft_drive/indra_apps/deft/input'
inpath = os.path.join(path, 'important')
outpath = os.path.join(path, 'fulltexts')
for fl in os.listdir(inpath):
    filename = os.fsdecode(fl)
    if filename.endswith('.pkl'):
        shortform = filename.split('_')[0]
        with open(os.path.join(inpath, fl), 'rb') as f:
            text_content = pickle.load(f)
        text_content = list(set(text_content.values()))
        plaintexts = get_plaintexts(text_content)
        with open(os.path.join(outpath, f'{shortform}_plaintexts.pkl'),
                  'wb') as f:
            pickle.dump(plaintexts, f)
