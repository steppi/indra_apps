from indra.tools import assemble_corpus as ac
import pandas as pd
import numpy as np
import networkx as nx
from copy import deepcopy
from get_complexes import get_dbrefs
import pickle
from fuzzywuzzy import fuzz
import seaborn as sb
from collections import defaultdict
from cachetools import cached
from cachetools.keys import hashkey
from itertools import combinations


class fuzz_set(object):
    def __init__(self, strings=set([]), cutoff=0.85):
        assert all(isinstance(x, str) for x in strings)
        self.__strings = set(strings)
        self.__cutoff = cutoff

    def __len__(self):
        return len(self.__string)

    def __contains__(self, value):
        for string in self.__strings:
            if fuzz_ratio(string, value) > self.__cutoff:
                return True
        else:
            return False

    def __repr__(self):
        return 'fuzz_set({})'.format(str([value for value in self.__strings]))

    def add(self, value):
        in_set = False
        for string in self.__strings:
            if fuzz_ratio(string, value) > self.__cutoff:
                if value < string:
                    self.__strings.remove(string)
                    self.__strings.add(value)
                in_set = True
        if not in_set:
            self.__strings.add(value)

    def remove(self, value):
        for string in self.__strings:
            if fuzz_ratio(string, value) > self.__cutoff:
                self.__strings.remove(string)

    def equiv(self, value):
        matches = [string for string in self.__strings
                   if fuzz_ratio(string, value) > self.__cutoff]
        if matches:
            return min(matches)
        else:
            raise ValueError


def get_ungrounded(stmts):
    return [stmt for stmt in stmts
            if {} in [member.db_refs for member in stmt.members]]


def add_grounding_map_groundings(stmts):
    output = []
    for stmt in stmts:
        new_stmt = deepcopy(stmt)
        for agent in new_stmt.members:
            if not agent.db_refs:
                grounding = ac.grounding_map.get(agent.name.upper())
                if grounding:
                    agent.db_refs = grounding
                    output.append(new_stmt)
    return output


mapping = {'polbeta': 'dna polymerase beta',
           'tfn': 'transferrin',
           'rxralpha': 'rxra',
           'ran': 'ara24',
           'aid': 'aicda',
           'signal transducer and_activator of transcription (stat)-3':
           'stat3',
           'amyloid precursor protein': 'app',
           'protein interacting with c kinase 1': 'pick1',
           'glyceraldehyde 3-phosphatase dehydrogenase': 'gapdh'}


def map_groundings(stmts, mapping):
    output = []
    for stmt in stmts:
        new_stmt = deepcopy(stmt)
        for agent in new_stmt.members:
            if not agent.db_refs:
                if agent.name in mapping:
                    agent.db_refs = get_dbrefs(mapping[agent.name])
        output.append(new_stmt)
    return output


@cached(cache={}, key=lambda text1, text2: hashkey(frozenset([text1, text2])))
def fuzz_ratio(text1, text2):
    """cached version of fuzzywuzzy.fuzz.ratio.
    """
    return fuzz.ratio(text1, text2)/100


def duplicate_ppi(ppi1, ppi2, cutoff=0.85):
    # sorry
    try:
        same_sentence = fuzz_ratio(ppi1[0], ppi2[0]) > cutoff
        same_source = (ppi1[1].evidence[0].source_api ==
                       ppi2[1].evidence[0].source_api)
        members1 = {tuple(sorted(agent.db_refs.items()))
                    for agent in ppi1[1].agent_list()}
        members2 = {tuple(sorted(agent.db_refs.items()))
                    for agent in ppi2[1].agent_list()}
        return same_sentence and (members1 == members2) and same_source
    except Exception:
        return False


def remove_duplicate_ppis(ppi_list, cutoff=0.85):
    newlist = []
    while ppi_list:
        first_ppi = deepcopy(ppi_list[0])
        newlist.append(first_ppi)
        filter_list = []
        for ppi in ppi_list:
            if not duplicate_ppi(first_ppi, ppi, cutoff):
                filter_list.append(deepcopy(ppi))
        ppi_list = filter_list
    return newlist


def filter_ungrounded(stmts):
    groundings = set(['HGNC', 'UP', 'IP', 'FPLX', 'PFAM',
                      'NXPFA', 'CHEBI', 'GO', 'MESH'])
    filtered = []
    for stmt in stmts:
        if not all(stmt.agent_list()):
            continue
        new_stmt = deepcopy(stmt)
        if all(groundings.intersection(agent.db_refs.keys())
               for agent in stmt.agent_list()):
            filtered.append(new_stmt)
    return filtered


# build mapping of pmids to statements
def get_pmid_mapping(stmts):
    def get_agent_id(agent):
        grounding_priority = ('HGNC', 'UP', 'IP', 'FPLX', 'PFAM',
                              'NXPFA', 'CHEBI', 'GO', 'MESH')
        for grounding in grounding_priority:
            db_refs = agent.db_refs.get(grounding)
            if db_refs:
                return '{}:{}'.format(grounding, db_refs)
    frame = []
    seen = set([])
    for stmt, truth_value in stmts:
        pmid = stmt.evidence[0].pmid
        sentence = stmt.evidence[0].text
        agent_ids = sorted([get_agent_id(agent)
                            for agent in stmt.agent_list()])
        agents_key = ':::'.join(agent_ids)
        reader = stmt.evidence[0].source_api
        key = (pmid, sentence, agents_key, reader, truth_value)
        if key not in seen:
            frame.append([pmid, agents_key, reader,
                          truth_value, sentence, stmt])
            seen.add(key)
    result = pd.DataFrame(frame, columns=['pmid', 'agents_key', 'reader',
                                          'type', 'sentence', 'stmt'])
    result.set_index(['pmid', 'agents_key', 'reader', 'type', 'sentence'],
                     inplace=True, drop=False)
    result.sort_index(inplace=True)
    return result


def deduplicate_mapping(mapping_df, cutoff=0.85):
    new_pmid_mapping_df = deepcopy(pmid_mapping_df)
    groups = mapping_df.groupby(level=['pmid', 'agents_key'])
    for _, new_df in groups:
        G = nx.Graph()
        for index, extraction in new_df.iterrows():
            G.add_node(index, sentence=extraction['sentence'])
        node_pairs = combinations(G.nodes(data=True), 2)
        for (index1, data1), (index2, data2) in node_pairs:
            sentence1, sentence2 = data1['sentence'], data2['sentence']
            ratio = fuzz_ratio(sentence1, sentence2)
            if ratio > cutoff:
                G.add_edge(index1, index2)
        for component in nx.connected_components(G):
            least_sentence = min(G.node[x]['sentence'] for x in component)
            indices = list(component)
            new_pmid_mapping_df.loc[indices, 'sentence'] = least_sentence
    duplicates = new_pmid_mapping_df.groupby(level=['pmid', 'agents_key',
                                                    'reader', 'sentence'])
    return duplicates.first()


def get_statements_table(mapping_df):
    groups = mapping_df.groupby(level=['pmid', 'agents_key', 'sentence'])
    frame = []
    for index1, new_df, in groups:
        new_row = {'index': index1}
        new_row['sentence'] = index1[-1]
        for index2, row in new_df.iterrows():
            if row.reader == 'nlm_ppi':
                if row['type']:
                    new_row['nlm_true'] = row.stmt
                else:
                    new_row['nlm_false'] = row.stmt
            else:
                new_row[row.reader] = row.stmt
        frame.append(new_row)
    output = pd.DataFrame(frame)
    return output.set_index('index').fillna('None')


# remove duplicate ppis in ppi mapping
def remove_duplicates_in_mapping(pmid_mapping):
    new_pmid_mapping = deepcopy(pmid_mapping)
    for pmid, extractions in new_pmid_mapping.items():
        extractions['true'] = remove_duplicate_ppis(extractions['true'])
        extractions['false'] = remove_duplicate_ppis(extractions['false'])
    return new_pmid_mapping


def get_counts_df(db_pmid_mapping, nlm_pmid_mapping):
    source_counts = {}
    for pmid, extractions in db_pmid_mapping.items():
        source_counts[pmid] = defaultdict(int)
        for stmt in extractions['true']:
            source_counts[pmid][stmt[1].evidence[0].source_api] += 1
    for pmid, extractions in nlm_pmid_mapping.items():
        if pmid not in source_counts:
            source_counts[pmid] = defaultdict(int)
        for stmt in extractions['true']:
            source_counts[pmid]['nlm_true'] += 1
        for stmt in extractions['false']:
            source_counts[pmid]['nlm_false'] += 1
    frame = []
    for pmid, counts in source_counts.items():
        frame.append([pmid, counts['reach'], counts['sparser'],
                      counts['nlm_true'], counts['nlm_false']])
    result = pd.DataFrame(frame, columns=['pmid', 'reach', 'sparser',
                                          'nlm_true', 'nlm_false'])
    result.index = result.pmid
    result = result.drop(['pmid'], axis=1)
    return result


nlm_true_stmts = ac.load_statements('../work/nlm_ppi_true_statements.pkl')
nlm_false_stmts = ac.load_statements('../work/nlm_ppi_false_statements.pkl')
nlm_filtered_true = filter_ungrounded(nlm_true_stmts)
nlm_filtered_false = filter_ungrounded(nlm_false_stmts)
nlm_stmts = [(stmt, True) for stmt in nlm_filtered_true]
nlm_stmts += [(stmt, False) for stmt in nlm_filtered_false]

with open('../work/db_interaction_stmts_by_pmid.pkl', 'rb') as f:
    db_pmid_mapping = pickle.load(f)
db_stmts = []
for pmid, extraction in db_pmid_mapping.items():
    for stmt in extraction:
        db_stmts.append(stmt)
db_stmts = filter_ungrounded(db_stmts)
db_stmts = [(stmt, True) for stmt in db_stmts]
stmts = nlm_stmts + db_stmts
pmid_mapping_df = get_pmid_mapping(stmts)
dedup = deduplicate_mapping(pmid_mapping_df)
stmts_table = get_statements_table(dedup)
weird = []
for stmt in nlm_filtered_true:
    agents = stmt.agent_list()
    for agent in agents:
        if not agent.db_refs.get('HGNC') and not agent.db_refs.get('FPLX'):
            weird.append((agent, agent.db_refs))
        

# weird = pmid_mapping['27879200']['false']
# weird_filtered = pmid_mapping_filtered['27879200']['false']

# weird = sorted(weird, key=lambda x: sorted(agent.name
#                                            for agent in x[1].agent_list()))
# weird_filtered = sorted(weird_filtered,
#                         key=lambda x: sorted(agent.name
#                                              for agent in x[1].agent_list()))


counts1 = [[str(pmid), len(extractions1['true']), len(extractions1['false']),
            len(extractions2['true']), len(extractions2['false'])]
           for (pmid, extractions1), (_, extractions2)
           in zip(pmid_mapping.items(),
                  pmid_mapping_filtered.items())]

counts1_df = pd.DataFrame(counts1, columns=['pmid', 'nlm_#true_ppi',
                                            'nlm_#false_ppi',
                                            '#nlm_true_no_dupes',
                                            '#nlm_false_no_dupes'])
                          

counts1_df.to_csv('../work/extraction_counts.tsv', sep='\t', index=False)



new_mapping = nlm_pmid_mapping
for pmid, extraction in db_pmid_mapping.items():
    if pmid not in new_mapping:
        new_mapping[pmid] = {'true': [], 'false': []}
        for stmt in extraction:
            if None not in stmt.agent_list():
                new_stmt = deepcopy(stmt)
                sentence = new_stmt.evidence[0].text
                new_mapping[pmid]['true'].append((sentence, new_stmt))

db_pmid_mapping_filtered = remove_duplicates_in_mapping(new_mapping)

# now we will build a multi-indexed data-frame

counts2 = [[str(pmid), len(extractions1['true']), len(extractions1['false']),
            len(extractions2['true']), len(extractions2['false'])]
           for (pmid, extractions1), (_, extractions2)
           in zip(new_mapping.items(),
                  db_pmid_mapping.items())]
counts_df2 = pd.DataFrame(counts2, columns=['pmid2', 'db_#true_ppi',
                                            'db_#false_ppi',
                                            'db_#true_no_dupes',
                                            'db_#false_no_dupes'])

