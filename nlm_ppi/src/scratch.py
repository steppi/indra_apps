from indra.tools import assemble_corpus as ac
import pandas as pd
import networkx as nx
from copy import deepcopy
from get_complexes import get_dbrefs
import pickle
from fuzzywuzzy import fuzz
from collections import defaultdict
from itertools import combinations


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


def filter_ungrounded(stmts):
    groundings = set(['HGNC', 'UP', 'IP', 'FPLX', 'PFAM',
                      'NXPFA', 'CHEBI', 'MESH'])
    filtered = []
    for stmt in stmts:
        if not all(stmt.agent_list()) or len(stmt.agent_list()) != 2:
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
                              'NXPFA', 'CHEBI', 'MESH')
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
            ratio = fuzz.ratio(sentence1, sentence2)/100
            if ratio > cutoff:
                G.add_edge(index1, index2)
        for component in nx.connected_components(G):
            least_sentence = min(G.node[x]['sentence'] for x in component)
            indices = list(component)
            new_pmid_mapping_df.loc[indices, 'sentence'] = least_sentence
    duplicates = new_pmid_mapping_df.groupby(by=['pmid', 'agents_key',
                                                 'reader', 'sentence'],
                                             as_index=False)
    return duplicates.first()


def get_statements_table(mapping_df):
    groups = mapping_df.groupby(by=['pmid', 'agents_key', 'sentence'],
                                as_index=False)
    frame = []
    for index1, new_df, in groups:
        new_row = {'reach': '', 'sparser': '',
                   'nlm_true': '', 'nlm_false': ''}
        for index2, row in new_df.iterrows():
            if row.reader == 'nlm_ppi':
                if row['type']:
                    new_row['nlm_true'] = row.stmt
                else:
                    new_row['nlm_false'] = row.stmt
            else:
                new_row[row.reader] = row.stmt
            new_row['pmid'] = row['pmid']
            new_row['sentence'] = row['sentence']
            new_row['agents_key'] = row['agents_key']
        frame.append(new_row)
    output = pd.DataFrame(frame)
    output = output[['pmid', 'sentence', 'agents_key', 'sparser',
                     'reach', 'nlm_true', 'nlm_false']]
    return output


nlm_true = ac.load_statements('../work/nlm_ppi_true_statements.pkl')
nlm_false = ac.load_statements('../work/nlm_ppi_false_statements.pkl')
nlm_filtered_true = filter_ungrounded(nlm_true)
nlm_filtered_false = filter_ungrounded(nlm_false)
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
dedup = deduplicate_mapping(pmid_mapping_df, cutoff=0.8)
stmts_table = get_statements_table(dedup)
stmts_table = stmts_table.sort_values(['pmid', 'sentence'])
stmts_table.to_csv('../work/stmts_table.tsv', sep='\t', index=False)

reach_nlm_agree = stmts_table[stmts_table.reach.astype('bool')
                              & stmts_table.nlm_true.astype('bool')]
reach_nlm_disagree = stmts_table[stmts_table.reach.astype('bool')
                                 & stmts_table.nlm_false.astype('bool')]
nlm_nlm_disagree = stmts_table[stmts_table.nlm_true.astype('bool')
                               & stmts_table.nlm_false.astype('bool')]
wtf = stmts_table[~stmts_table.nlm_true.astype('bool') &
                  ~stmts_table.nlm_false.astype('bool')]

