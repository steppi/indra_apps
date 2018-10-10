import pandas as pd
import re
import json
from indra_db.util import get_primary_db
from indra_db.client import get_reader_output
from collections import defaultdict
from itertools import groupby
import random
import pickle


db = get_primary_db()


def reader_info(pmid):
    """Given a pmid, get sentences and sentence to entity list mappings
    from the associated reach JSONS.

    Output is of the form
    {'sentences': {'sentence_id': tuple(sentence texts)},
    'entities': {'sentence_id': frozenset(frozenset(tuple(groundings)))...}
    """
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


pmid_mapper = pd.read_pickle("../work/extractions_table.pkl")
pmids = set(pmid_mapper.pmid.unique())


nlm_sents = pmid_mapper[pmid_mapper.reader == 'nlm_ppi']
nlm_sents = nlm_sents.groupby('pmid').sentence.unique()
nlm_sents = nlm_sents.reset_index()


reach_sents = pmid_mapper[pmid_mapper.reader == 'reach']
reach_sents = reach_sents.groupby('pmid').sentence.unique()
reach_sents = reach_sents.reset_index()

mapper = {}
for pmid in pmids:
    mapper[pmid] = reader_info(pmid)

sents = {}
for pmid in pmids:
    sentences = mapper[pmid]['sentences'].items()
    sentences = sorted(sentences, key=lambda x: x[1])
    sentences = tuple(random.choice(tuple(group)) for _, group in
                      groupby(sentences, key=lambda x: x[1]))
    sents[pmid] = sentences

ents = {}
for pmid in pmids:
    entities = mapper[pmid]['entities'].items()
    for key, value in entities:
        ents[key] = value

with open('../work/all_reach_sentences.pkl', 'wb') as f:
    pickle.dump(sents, f)


with open('../work/all_reach_groundings.pkl', 'wb') as f:
    pickle.dump(ents, f)
