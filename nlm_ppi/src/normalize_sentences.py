from indra.tools import assemble_corpus as ac
import pandas as pd
import networkx as nx
from copy import deepcopy
import pickle
from fuzzywuzzy import fuzz
from itertools import combinations
import regex
import random


def filter_none(stmts):
    filtered = []
    for stmt in stmts:
        if all(stmt.agent_list()) and len(stmt.agent_list()) == 2:
            filtered.append(deepcopy(stmt))
    return filtered


def sentence_match(sen1, sen2, cutoff1=0.8, cutoff2=0.5):
    ratio = fuzz.ratio(sen1, sen2)
    if ratio > cutoff1:
        return True
    elif ratio > cutoff2:
        if ' xref ' in sen1 and ' xref ' not in sen2:
            reg1 = sen1.replace(' xref ', '.+')
            # convert between ratio cutoff and number of errors to allow
            e = len(sen1 + sen2) - cutoff1*len(sen1 + sen2)
            r = regex.compile(reg1 + '\{{}\}'.format(e))
            return r.match(sen2)
        elif ' xref ' in sen2 and ' xref ' not in sen1:
            sen1, sen2 = sen2, sen1
            reg1 = sen1.replace(' xref ', '.+')
            # convert between ratio cutoff and number of errors to allow
            e = len(sen1 + sen2) - cutoff1*len(sen1 + sen2)
            r = regex.compile(reg1 + '\{{}\}'.format(e))
            return r.match(sen2)
    else:
        return False


def get_pmid_mapping(stmts):
    frame = []
    seen = set([])
    for stmt, truth_value in stmts:
        pmid = stmt.evidence[0].pmid
        sentence = stmt.evidence[0].text
        agents = frozenset([frozenset(agent.db_refs.items())
                            for agent in stmt.agent_list()])
        reader = stmt.evidence[0].source_api
        key = (pmid, sentence, agents, reader, truth_value)
        if key not in seen:
            frame.append([pmid, agents, reader,
                          truth_value, sentence, stmt])
            seen.add(key)
    result = pd.DataFrame(frame, columns=['pmid', 'agents', 'reader',
                                          'type', 'sentence', 'stmt'])
    result.set_index(['pmid', 'agents', 'reader', 'type', 'sentence'],
                     inplace=True, drop=False)
    result.sort_index(inplace=True)
    return result


def deduplicate_mapping(mapping_df, cutoff1=0.8, cutoff2=0.5):
    new_mapping_df = deepcopy(mapping_df)
    new_mapping_df['normed_sentence'] = new_mapping_df['sentence']
    groups = mapping_df.groupby(level=['pmid'])
    for _, new_df in groups:
        G = nx.Graph()
        for index, extraction in new_df.iterrows():
            G.add_node(index, sentence=extraction['sentence'])
        node_pairs = combinations(G.nodes(data=True), 2)
        for (index1, data1), (index2, data2) in node_pairs:
            sentence1, sentence2 = data1['sentence'], data2['sentence']
            if sentence_match(sentence1, sentence2):
                G.add_edge(index1, index2)
        for component in nx.connected_components(G):
            least_sentence = min(G.node[x]['sentence'] for x in component)
            indices = list(component)
            new_mapping_df.loc[indices, 'normed_sentence'] = least_sentence
    new_mapping_df['sentence_id'] = \
        new_mapping_df['normed_sentence'].apply(lambda x: hash(x))
    new_mapping_df = new_mapping_df.drop(['normed_sentence'], axis=1)
    return new_mapping_df


if __name__ == '__main__':
    nlm_true = ac.load_statements('../work/nlm_ppi_true_statements.pkl')
    nlm_false = ac.load_statements('../work/nlm_ppi_false_statements.pkl')
    with open('../work/db_interaction_stmts_by_pmid.pkl', 'rb') as f:
        db_pmid_mapping = pickle.load(f)
    db_stmts = []
    for pmid, extraction in db_pmid_mapping.items():
        for stmt in extraction:
            db_stmts.append(stmt)
    nlm_true = filter_none(nlm_true)
    nlm_false = filter_none(nlm_false)
    db_stmts = filter_none(db_stmts)

    nlm_stmts = [(stmt, True) for stmt in nlm_true]
    nlm_stmts += [(stmt, False) for stmt in nlm_false]
    db_stmts = [(stmt, True) for stmt in db_stmts]
    stmts = nlm_stmts + db_stmts

    mapping_df = get_pmid_mapping(stmts)
    dedup = deduplicate_mapping(mapping_df, cutoff1=0.8, cutoff2=0.5)
    dedup.to_pickle('../work/extractions_table.pkl')
