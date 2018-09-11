from indra.tools import assemble_corpus as ac
import pandas as pd
import numpy as np
from copy import deepcopy
from get_complexes import get_dbrefs
import pickle

true_stmts = ac.load_statements('../work/nlm_ppi_true_statements.pkl')
false_stmts = ac.load_statements('../work/nlm_ppi_false_statements.pkl')

ungrounded = [agent for stmt in true_stmts + false_stmts
              for agent in stmt.members if not agent.db_refs]
ungrounded_names = [agent.name for agent in ungrounded]
ungrounded_names = pd.Series(np.array(ungrounded_names))
ungrounded_counts = ungrounded_names.value_counts()
unique_ungrounded = list(set(ungrounded_names))


h2b_groundings = get_dbrefs('h2b')
h2b_groundings2 = get_dbrefs('Histone H2B')


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


true_stmts = add_grounding_map_groundings(true_stmts)
false_stmts = add_grounding_map_groundings(false_stmts)

true_stmts = map_groundings(true_stmts, mapping)
false_stmts = map_groundings(false_stmts, mapping)

pmid_mapping = {}
for stmt in true_stmts:
    pmid = stmt.evidence[0].pmid
    if pmid not in pmid_mapping:
        pmid_mapping[pmid] = {'true': [], 'false': []}
    pmid_mapping[pmid]['true'].append(stmt)
for stmt in false_stmts:
    pmid = stmt.evidence[0].pmid
    if pmid not in pmid_mapping:
        pmid_mapping[pmid] = {'true': [], 'false': []}
    pmid_mapping[pmid]['false'].append(stmt)




ac.dump_statements(true_stmts, '../work/nlm_ppi_true_statements.pkl')
ac.dump_statements(false_stmts, '../work/nlm_ppi_false_statements.pkl')
