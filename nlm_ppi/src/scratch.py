from indra.tools import assemble_corpus as ac
import pandas as pd
import networkx as nx
from copy import deepcopy
import regex
import pickle
from fuzzywuzzy import fuzz
from itertools import combinations
from get_complexes import get_dbrefs
from cachetools import cached, LRUCache
from cachetools.keys import hashkey
from collections import Counter



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


@cached(LRUCache(maxsize=500), key=lambda x:
        tuple(sorted([agent.name for agent
                      in x.agent_list()])))
def translation_map(stmt):
    mapper = {'polbeta': 'dna polymerase beta',
              'tfn': 'transferrin',
              'rxralpha': 'rxra',
              'ran': 'ara24',
              'aid': 'aicda',
              'signal transducer and_activator of transcription (stat)-3':
              'stat3',
              'amyloid precursor protein': 'app',
              'protein interacting with c kinase 1': 'pick1',
              'glyceraldehyde 3-phosphatase dehydrogenase': 'gapdh',
              'upar': 'urokinase-type plasminogen activator'}
    new_stmt = deepcopy(stmt)
    for agent in new_stmt.agent_list():
        if agent.name in mapper:
            new_ground = get_dbrefs(mapper[agent.name])
            agent.db_refs = new_ground
            agent.db_refs['text'] = agent.name
    return new_stmt


def is_grounded(stmt):
    groundings = set(['HGNC', 'UP', 'IP', 'FPLX', 'PFAM',
                      'NXPFA', 'CHEBI'])
    return all(groundings.intersection(agent.db_refs.keys())
               for agent in stmt.agent_list())


def get_statements_table(mapping_df):
    groups = mapping_df.groupby(by=['pmid', 'agents', 'sentence'],
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
            new_row['agents'] = row['agents']
        frame.append(new_row)
    output = pd.DataFrame(frame)
    output = output[['pmid', 'sentence', 'agents', 'sparser',
                     'reach', 'nlm_true', 'nlm_false']]
    return output


def equiv_agent(agent1, agent2):
    return not agent1.db_refs.items().isdisjoint(agent2.db_refs.items())


def nlm_db_grounding_map(db_stmts, nlm_stmts):
    mapper = {}
    G = nx.Graph()
    db_groundings = list(set([frozenset(agent.db_refs.items())
                             for stmt in db_stmts
                              for agent in stmt.agent_list()]))
    nlm_groundings = list(set([frozenset(agent.db_refs.items())
                              for stmt in nlm_stmts
                              for agent in stmt.agent_list()]))
    G.add_nodes_from([(grounding, {'source': 'db'})
                      for grounding in db_groundings])
    G.add_nodes_from([(grounding, {'source': 'nlm'})
                      for grounding in nlm_groundings])
    node_pairs = combinations(G.nodes(data=True), 2)
    for (grounding1, _), (grounding2, _) in node_pairs:
        lower1 = frozenset([(key, value.lower())
                            for key, value in grounding1])
        lower2 = frozenset([(key, value.lower())
                            for key, value in grounding2])
        if not lower1.isdisjoint(lower2):
            G.add_edge(grounding1, grounding2)
    for component in nx.connected_components(G):
        db_groundings = [grounding for grounding in component
                         if G.node[grounding]['source'] == 'db']
        if db_groundings:
            canonical = max(db_groundings,
                            key=lambda x: len(x))
        else:
            canonical = max(component,
                            key=lambda x: len(x))
        for grounding in component:
            mapper[grounding] = dict(canonical)
    return mapper


def fix_grounding(row, mapper):
    new_stmt = deepcopy(row['stmt'])
    for agent in new_stmt.agent_list():
        grounding = frozenset(agent.db_refs.items())
        if grounding in mapper:
            agent.db_refs = mapper[grounding]
    agents = frozenset([frozenset(agent.db_refs.items()) for agent in
                       new_stmt.agent_list()])
    return agents


def get_mismatched(stmts_table):
    no_nlm = stmts_table[~stmts_table.nlm_true.astype('bool') &
                         ~stmts_table.nlm_false.astype('bool')]
    no_nlm = no_nlm.drop(['nlm_true', 'nlm_false'], axis=1)

    with_nlm = stmts_table[stmts_table.nlm_true.astype('bool') |
                           stmts_table.nlm_false.astype('bool')]
    with_nlm = with_nlm.drop(['reach', 'sparser'], axis=1)
    mismatched = pd.merge(no_nlm, with_nlm, on=['pmid', 'sentence'])
    groups = mismatched.groupby(['pmid', 'sentence'])
    frame = []
    for (pmid, sentence), new_df in groups:
        db = []
        nlm = []
        new_row = {'pmid': pmid, 'sentence': sentence,
                   'db_only': [], 'nlm_only': []}
        for _, row in new_df.iterrows():
            for reader in ['reach', 'sparser', 'nlm_true', 'nlm_false']:
                if row[reader]:
                    groundings = [frozenset(agent.db_refs.items())
                                  for agent in row[reader].agent_list()]
                    if reader in ('reach', 'sparser'):
                        db.extend(groundings)
                    else:
                        nlm.extend(groundings)
            new_row['db_only'] = set(db) - set(nlm)
            new_row['nlm_only'] = set(nlm) - set(db)
        frame.append(new_row)
    output = pd.DataFrame(frame)
    return output[['pmid', 'sentence', 'db_only', 'nlm_only']]


pmid_mapping = pd.read_pickle('../work/pmid_stmt_table.pkl')
# pmid_mapping['stmt'] = pmid_mapping['stmt'].apply(lambda x:
#                                                   translation_map(x))

nlm_stmts = pmid_mapping[pmid_mapping['reader'] == 'nlm_ppi']['stmt'].values
db_stmts = pmid_mapping[~(pmid_mapping['reader'] == 'nlm_ppi')]['stmt'].values

mapper = nlm_db_grounding_map(db_stmts, nlm_stmts)
pmid_mapping['agents'] = pmid_mapping.apply(lambda x:
                                            fix_grounding(x, mapper),
                                            axis=1)
pmid_mapping = pmid_mapping[pmid_mapping['stmt'].apply(is_grounded)]
stmts_table = get_statements_table(pmid_mapping)
stmts_table = stmts_table.sort_values(['pmid', 'sentence'])

mismatched = get_mismatched(stmts_table)
db_only = [agent for row in mismatched.db_only.values for agent in row]
nlm_only = [agent for row in mismatched.nlm_only.values
            for agent in row]
unique_db_only = list(set(db_only))
unique_nlm_only = list(set(nlm_only))
stmts_table.replace('', 'None').to_csv('../work/stmts_table.tsv',
                                       sep='\t', index=False)


reach_nlm_agree = stmts_table[stmts_table.reach.astype('bool')
                              & stmts_table.nlm_true.astype('bool')]
reach_nlm_disagree = stmts_table[stmts_table.reach.astype('bool')
                                 & stmts_table.nlm_false.astype('bool')]
nlm_nlm_disagree = stmts_table[stmts_table.nlm_true.astype('bool')
                               & stmts_table.nlm_false.astype('bool')]

no_nlm = stmts_table[~(stmts_table.nlm_true.astype('bool') |
                       stmts_table.nlm_false.astype('bool'))]
no_nlm = no_nlm.drop(['nlm_true', 'nlm_false'], axis=1)

with_db = stmts_table[stmts_table.reach.astype('bool') |
                      stmts_table.sparser.astype('bool')]
with_db.replace('', 'None').to_csv('../work/db_stmts_table.tsv',
                                   sep='\t', index=False)
with_nlm = stmts_table[stmts_table.nlm_true.astype('bool') |
                       stmts_table.nlm_false.astype('bool')]
with_nlm.replace('', 'None').to_csv('../work/nlm_stmts_table.tsv',
                                    sep='\t', index=False)

reach_and_nlm = stmts_table[stmts_table.reach.astype('bool') &
                            (stmts_table.nlm_true.astype('bool') |
                            stmts_table.nlm_false.astype('bool'))]



reach_nlm_agree = with_nlm[with_nlm.reach.astype('bool')
                           & (with_nlm.nlm_true.astype('bool') |
                           ~with_nlm.reach.astype('bool') &
                           with_nlm.nlm_false.astype('bool'))]
reach_nlm_disagree1 = with_nlm[with_nlm.reach.astype('bool')
                               & with_nlm.nlm_false.astype('bool')]
reach_nlm_disagree1[['sentence', 'reach']].to_csv('../work/disagreement_table.tsv',
                                                  sep='\t', index=False)
reach_nlm_disagree2 = with_nlm[~with_nlm.reach.astype('bool') &
                               with_nlm.nlm_true.astype('bool')]

x = with_nlm[['reach', 'sparser', 'nlm_true', 'nlm_false']].astype('bool')
x.groupby('reach')['nlm_true'].value_counts()/x.groupby('reach')['nlm_true'].count()
with_nlm[with_nlm.astype('bool')['nlm_true']][['sentence', 'nlm_true']].to_csv('../work/nlm_stmts_table.tsv', sep='\t')


db_agents = [agent for stmt in stmts_table.reach if stmt
             for agent in stmt.agent_list()]
db_agents.extend([agent for stmt in stmts_table.sparser if stmt
                  for agent in stmt.agent_list()])
nlm_agents = [agent for stmt in stmts_table.nlm_true if stmt
              for agent in stmt.agent_list()]
nlm_agents.extend([agent for stmt in stmts_table.nlm_false if stmt
                   for agent in stmt.agent_list()])
db_groundings = [frozenset(agent.db_refs.items()) for agent in db_agents]
nlm_groundings = [frozenset(agent.db_refs.items()) for agent in nlm_agents]
in_both = set(db_groundings) & set(nlm_groundings)
a = Counter(db_groundings)
b = Counter(nlm_groundings)


x = stmts_table[stmts_table['agents'].apply(lambda z: z <= in_both)]
with_nlm_true = x[x.nlm_true.astype('bool')]
with_nlm_false = x[x.nlm_false.astype('bool')]
