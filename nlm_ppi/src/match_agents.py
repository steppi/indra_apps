import pandas as pd
import numpy as np
import pickle
import networkx as nx

extr_df = pd.read_pickle('../work/extractions_table5.pkl')
with open('../work/all_reach_groundings.pkl', 'rb') as f:
    reach_ents = pickle.load(f)

fplx_map = pd.read_csv('~/indra/indra/resources/famplex_map.tsv',
                       sep='\t', names=['type', 'grounding', 'FPLX'])
uniprot_map = pd.read_csv('~/indra/indra/resources/uniprot_entries.tsv',
                          sep='\t',
                          dtype=object)
hgnc_map = pd.read_csv('~/indra/indra/resources/hgnc_entries.tsv',
                       sep='\t',
                       names=['HGNC_ID', 'Approved_Symbol',
                              'Approved_Name', 'Status',
                              'Synonyms', 'Entrez_Gene_ID',
                              'UniProt_ID', 'MouseGenomeDB_ID',
                              'RatGenomeDB_ID'])

key_mapper = {'uniprot': 'UP',
              'be': 'FPLX',
              'interpro': 'IP',
              'pfam': 'PF',
              'uaz': 'uaz'}

grounding_set = set([])
not_in_fplx = set([])
not_in_hgnc = set([])
grounding_types = set([])
# as a warmup let's get all groundings missing from Famplex
for sentence, groundings in reach_ents.items():
    for grounding in groundings:
        temp = set()
        temp.add(('TEXT', grounding[0]))
        for key, value in grounding[1]:
            new_key = key_mapper[key]
            temp.add((new_key, value))
            if new_key == 'PF' or new_key == 'IP':
                print(new_key, value)
                fplx = fplx_map[fplx_map.type == new_key]
                fplx = fplx[fplx.grounding == value]
                print(fplx)
                if len(fplx) == 0:
                    not_in_fplx.add(grounding)
                else:
                    temp.add(('FPLX', fplx.FPLX.values[0]))
        grounding_set.add(frozenset(temp))


frame = []
for grounding in not_in_fplx:
    text = grounding[0]
    for name_space, entry in grounding[1]:
        frame.append({'name_space': key_mapper[name_space],
                      'text': text,
                      'entry': entry})
not_in_fplx_df = pd.DataFrame(frame)
not_in_fplx_df = not_in_fplx_df[['name_space', 'entry', 'text']]
not_in_fplx_df.sort_values(by=['name_space', 'entry'], inplace=True)
not_in_fplx_df = not_in_fplx_df.groupby(['name_space',
                                         'entry'],
                                        as_index=False).agg(lambda x:
                                                            ','.join(list(x)))

not_in_fplx_df.to_csv('../result/fplx_missing_entries.txt', sep='\t',
                      index=False)


