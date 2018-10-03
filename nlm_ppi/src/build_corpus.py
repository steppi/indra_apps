import pandas as pd
import numpy as np
import re
import json
from indra_db.util import get_primary_db
from indra_db.client import get_reader_output
from collections import defaultdict
from fuzzywuzzy import fuzz
import pickle


db = get_primary_db()


def reader_info(pmid):
    content = get_reader_output(db, pmid,
                                ref_type='pmid', reader='reach')
    reach_jsons = [json.loads(value['REACH'][0])
                   for _, value in content.items()]
    sentences = [sentence for reach_json in reach_jsons
                 for sentence in reach_json['sentences']['frames']]
    entities = [entity for reach_json in reach_jsons
                for entity in reach_json['entities']['frames']]

    sentence_map = {sentence['frame-id']: sentence['text']
                    for sentence in sentences
                    if re.match(r'^sent', sentence['frame-id'])}

    entity_map = defaultdict(list)
    for entity in entities:
        if 'sentence' in entity and entity.get('type') \
           in ['protein', 'family']:
            xrefs = entity.get('xrefs') if entity.get('xrefs') else []
            alt_xrefs = (entity.get('alt-xrefs')
                         if entity.get('alt-xrefs') else [])
            refs = [(ref['namespace'], ref['id']) for ref in xrefs + alt_xrefs]
            refs = frozenset(refs)
            entity_map[entity.get('sentence')].append((entity.get('text'),
                                                       refs))

    return {'sentences': sentence_map, 'entities': entity_map}


def fuzz_matrix(sent_list1, sent_list2):
    output = np.zeros((len(sent_list1), len(sent_list2)))
    for index1, sent1 in enumerate(sent_list1):
        for index2, sent2 in enumerate(sent_list2):
            output[index1, index2] = fuzz.ratio(sent1, sent2)/100
    return output


pmid_mapper = pd.read_pickle("../work/extractions_table.pkl")
pmids = set(pmid_mapper.pmid.unique())
pmid_blacklist = set(['24141421', '28855251', '25822970'])
pmids = pmids - pmid_blacklist

nlm_sents = pmid_mapper[pmid_mapper.reader == 'nlm_ppi']
nlm_sents = nlm_sents[~pmid_mapper.pmid.isin(pmid_blacklist)]
nlm_sents = nlm_sents.groupby('pmid').sentence.unique()
nlm_sents = nlm_sents.reset_index()
sent_counts = nlm_sents.sentence.apply(lambda x: len(x))

reach_sents = pmid_mapper[pmid_mapper.reader == 'reach']
reach_sents = reach_sents[~pmid_mapper.pmid.isin(pmid_blacklist)]
reach_sents = reach_sents.groupby('pmid').sentence.unique()
reach_sents = reach_sents.reset_index()

mapper = {}
for pmid in pmids:
    mapper[pmid] = reader_info(pmid)

db_sent_counts = {key: len(value['sentences'].keys())
                  for key, value in mapper.items()}
sent_counts['nlm'] = sent_counts['pmid'].apply(lambda x: db_sent_counts[x])

sents1 = nlm_sents[nlm_sents.pmid == '11266465'].sentence.values[0]
sents2 = {pmid: tuple(set(mapper[pmid]['sentences'].values()))
          for pmid in pmids}

with open('../work/all_reach_sentences.pkl', 'wb') as f:
    pickle.dump(sents2, f)

rats1 = fuzz_matrix(nlm_sents[nlm_sents.pmid == '11266465'].sentence.values[0],
                    mapper['11266465']['sentences'].values())

rats2 = fuzz_matrix(reach_sents[reach_sents.pmid ==
                                '11266465'].sentence.values[0],
                    mapper['11266465']['sentences'].values()
