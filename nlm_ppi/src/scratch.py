from indra.tools import assemble_corpus as ac
import pandas as pd
import numpy as np
from copy import deepcopy
from get_complexes import get_dbrefs
import pickle
from fuzzywuzzy import fuzz
import seaborn as sb
from collections import defaultdict


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


wat = []


def duplicate_ppi(ppi1, ppi2, cutoff=0.85):
    # sorry
    try:
        same_sentence = fuzz.ratio(ppi1[0], ppi2[0]) > cutoff
        same_source = (ppi1[1].evidence[0].source_api ==
                       ppi2[1].evidence[0].source_api)
        members1 = {tuple(sorted(agent.db_refs.items()))
                    for agent in ppi1[1].agent_list()}
        members2 = {tuple(sorted(agent.db_refs.items()))
                    for agent in ppi2[1].agent_list()}
        return same_sentence and (members1 == members2) and same_source
    except Exception:
        wat.append((deepcopy(ppi1), deepcopy(ppi2)))
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


# true_stmts = add_grounding_map_groundings(true_stmts)
# false_stmts = add_grounding_map_groundings(false_stmts)

# true_stmts = map_groundings(true_stmts, mapping)
# false_stmts = map_groundings(false_stmts, mapping)

# filter ungrounded statements. keep only hgnc or famplex groundings
def filter_ungrounded(stmts):
    filtered = []
    for stmt in true_stmts:
        new_stmt = deepcopy(stmt)
        for agent in new_stmt.agent_list():
            agent.db_refs = {key: value for key, value in agent.db_refs.items()
                             if key == 'HGNC' or key == 'FMPLX'}
        if {} not in [agent.db_refs for agent in stmt.agent_list()]:
            filtered.append(new_stmt)
    return filtered


# build mapping of pmids to statements
def get_pmid_mapping(stmt_dict):
    pmid_mapping = {}
    for stmt in stmt_dict['true']:
        pmid = stmt.evidence[0].pmid
        sentence = stmt.evidence[0].text
        if pmid not in pmid_mapping:
            pmid_mapping[pmid] = {'true': [], 'false': []}
        pmid_mapping[pmid]['true'].append((sentence, stmt))
    for stmt in stmt_dict['false']:
        pmid = stmt.evidence[0].pmid
        sentence = stmt.evidence[0].text
        if pmid not in pmid_mapping:
            pmid_mapping[pmid] = {'true': [], 'false': []}
        pmid_mapping[pmid]['false'].append((sentence, stmt))
    return pmid_mapping


# remove duplicate ppis in ppi mapping
def remove_duplicates_in_mapping(pmid_mapping):
    new_pmid_mapping = deepcopy(pmid_mapping)
    for pmid, extractions in new_pmid_mapping.items():
        extractions['true'] = remove_duplicate_ppis(extractions['true'])
        extractions['false'] = remove_duplicate_ppis(extractions['false'])
    return new_pmid_mapping


true_stmts = ac.load_statements('../work/nlm_ppi_true_statements.pkl')
false_stmts = ac.load_statements('../work/nlm_ppi_false_statements.pkl')
filtered_true = filter_ungrounded(true_stmts)
filtered_false = filter_ungrounded(false_stmts)
stmt_dict = {'true': true_stmts, 'false': false_stmts}

pmid_mapping = get_pmid_mapping(stmt_dict)
with open('nlm_ppi_stmts_by_pmid.pkl', 'rb') as f:
    pmid_mapping_filtered = pickle.load(f)

weird = pmid_mapping['27879200']['false']
weird_filtered = pmid_mapping_filtered['27879200']['false']

weird = sorted(weird, key=lambda x: sorted(agent.name
                                           for agent in x[1].agent_list()))
weird_filtered = sorted(weird_filtered,
                        key=lambda x: sorted(agent.name
                                             for agent in x[1].agent_list()))


counts1 = [[pmid, len(extractions1['true']), len(extractions1['false']),
            len(extractions2['true']), len(extractions2['false'])]
           for (pmid, extractions1), (_, extractions2)
           in zip(pmid_mapping.items(),
                  pmid_mapping_filtered.items())]

counts_df = pd.DataFrame(counts1, columns=['pmid', 'nlm_#true_ppi',
                                           'nlm_#false_ppi',
                                           '#nlm_true_no_dupes',
                                           '#nlm_false_no_dupes'],
                         dtype='str')

counts_df.to_csv('../work/extraction_counts.tsv', sep='\t', index=False)


with open('../work/db_interaction_stmts_by_pmid.pkl', 'rb') as f:
    db_pmid_mapping = pickle.load(f)

new_mapping = {}
for pmid, extraction in db_pmid_mapping.items():
    new_mapping[pmid] = {'true': [], 'false': []}
    for stmt in extraction:
        if None not in [agent for agent in stmt.agent_list()]:
            new_stmt = deepcopy(stmt)
            sentence = new_stmt.evidence[0].text
            new_mapping[pmid]['true'].append((sentence, new_stmt))


new_mapping_filtered = remove_duplicates_in_mapping(new_mapping)
counts2 = [[pmid, len(extractions1['true']), len(extractions1['false']),
            len(extractions2['true']), len(extractions2['false'])]
           for (pmid, extractions1), (_, extractions2)
           in zip(new_mapping.items(),
                  new_mapping_filtered.items())]
counts_df2 = pd.DataFrame(counts2, columns=['pmid2', 'db_#true_ppi',
                                            'db_#false_ppi',
                                            'db_#true_no_dupes',
                                            'db_#false_no_dupes'])

counts_df2 = counts_df2.drop(['db_#false_ppi', 'db_#false_no_dupes'], axis=1)
counts_df2.to_csv("../work/db_extraction_counts.tsv", sep='\t',
                  index=False)
all_counts = pd.concat([counts_df, counts_df2], axis=1)
result = all_counts[['pmid', '#nlm_true_no_dupes',
                     'db_#true_no_dupes']]
result.columns = ['pmid', 'nlm_extractions', 'db_extractions']
result.to_csv('../work/nlm_vs_db_extractions.tsv', sep='\t', index=False)

source_counts = {}
for pmid, extractions in new_mapping_filtered.items():
    source_counts[pmid] = defaultdict(int)
    for stmt in extractions['true']:
        source_counts[pmid][stmt[1].evidence[0].source_api] += 1

frame = []
for pmid, counts in source_counts.items():
    frame.append([pmid, counts['reach'], counts['sparser']])

update_result = pd.DataFrame(frame, columns=['pmid', 'reach', 'sparser'])
update_result['nlm'] = result['nlm_extractions']
update_result.to_csv("../work/nlm_vs_reach_vs_sparser_extractions.tsv",
                     sep="\t", index=False)

update_result.index = update_result.pmid
update_result = update_result.drop(['pmid'], axis=1)
update_result['nlm'] = pd.to_numeric(update_result['nlm'])
