import pandas as pd
import networkx as nx
from copy import deepcopy
from itertools import combinations
from get_complexes import get_dbrefs
from cachetools import cached, LRUCache
from collections import Counter
from indra.databases.hgnc_client import get_hgnc_name


# many apologies to anyone who has to use this or modify it, including
# future albert.

@cached(LRUCache(maxsize=500), key=lambda x:
        tuple(sorted([agent.name for agent
                      in x.agent_list()])))
def translation_map(stmt):
    """This is for matching groundings manually."""
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
            agent.db_refs['TEXT'] = agent.name
    return new_stmt


def get_statements_df(extractions_df):
    """Alternative format for extractions
    the columns are
    pmid sentence_id sentence agents sparser reach nlm_true nlm_false
    where the columns corresponding to the readers contain actual the
    actual indra statements that were extracted"""
    groups = extractions_df.groupby(by=['pmid', 'agents', 'sentence_id'],
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
            new_row['sentence_id'] = row['sentence_id']
            frame.append(new_row)
    output = pd.DataFrame(frame)
    output = output[['pmid', 'sentence_id', 'sentence', 'agents',
                     'sparser', 'reach', 'nlm_true', 'nlm_false']]
    return output


def equiv_agent(agent1, agent2):
    """We consider two agents to be equivalent if they have at least one
    shared entry in their db_refs"""
    return not agent1.db_refs.items().isdisjoint(agent2.db_refs.items())


def nlm_db_grounding_map(stmts):
    """Match groundings for a list of statements. Builds a graph of whose
    nodes are agents in the stmts. Edges are placed between agents that are
    equivalent in the sense of having a shared entry in their db_refs. Agents
    in the connected components of this graph are matched with each other.
    """
    mapper = {}
    G = nx.Graph()
    all_groundings = list(set([frozenset(agent.db_refs.items())
                               for stmt in stmts
                               for agent in stmt.agent_list()]))
    G.add_nodes_from(all_groundings)
    node_pairs = combinations(G.nodes(data=True), 2)
    for (grounding1, _), (grounding2, _) in node_pairs:
        lower1 = frozenset([(key, value.lower())
                            for key, value in grounding1])
        lower2 = frozenset([(key, value.lower())
                            for key, value in grounding2])
        if not lower1.isdisjoint(lower2):
            G.add_edge(grounding1, grounding2)
    for component in nx.connected_components(G):
        canonical = max(list(component),
                        key=lambda x: len(x))
        for grounding in component:
            mapper[grounding] = dict(canonical)
    return mapper


def fix_grounding(row, mapper):
    """Use the above grounding map to fix groundings in the table frame.
    For use in a pd.apply
    """
    new_stmt = deepcopy(row['stmt'])
    for agent in new_stmt.agent_list():
        grounding = frozenset(agent.db_refs.items())
        if grounding in mapper:
            agent.db_refs = mapper[grounding]
    agents = frozenset([frozenset(agent.db_refs.items()) for agent in
                       new_stmt.agent_list()])
    return agents


# Exclude pmids with a stupid number of extractions due to the inclusion of
# tables and other things
pmid_blacklist = set(['24141421', '28855251', '25822970'])

# read in extracts_df
extractions_df = pd.read_pickle('../work/extractions_table.pkl')

# this section is for matching groundings. pull out statements
# build grounding mapper and apply it dataframe. agents key is a
# frozenset of frozenset of canonical agent.db_refs.items()
nlm_stmts = extractions_df[extractions_df['reader'] ==
                           'nlm_ppi']['stmt'].values

db_stmts = extractions_df[~(extractions_df['reader'] ==
                            'nlm_ppi')]['stmt'].values

mapper = nlm_db_grounding_map(list(nlm_stmts) + list(db_stmts))
extractions_df['agents'] = extractions_df.apply(lambda x:
                                                fix_grounding(x, mapper),
                                                axis=1)


# get the names of agents from agent keys. one off for use in pd.apply
# add agent names to extractions df
def __get_agent_names(row):
    return pd.Series(sorted(dict(agent)['TEXT'] if
                            'TEXT' in dict(agent) else
                            (get_hgnc_name(dict(agent)['HGNC'])
                            if 'HGNC' in dict(agent)
                             else '')
                            for agent in row.agents))


extractions_df[['agent1',
                'agent2']] = extractions_df.apply(__get_agent_names,
                                                  axis=1)

stmts_table = get_statements_df(extractions_df)
stmts_table = stmts_table.sort_values(['pmid', 'sentence_id'])

# this section is for identifying agents that are recognized by both
# reach and nlm. sorry to anyone who has to work with this. including
# future albert

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
reach_nlm_disagree1[['sentence',
                     'reach']].to_csv('../work/disagreement_table.tsv',
                                      sep='\t', index=False)
reach_nlm_disagree2 = with_nlm[~with_nlm.reach.astype('bool') &
                               with_nlm.nlm_true.astype('bool')]

db_groundings = [agent for _, row in stmts_table.iterrows()
                 if row.reach or row.sparser
                 for agent in row['agents']]

nlm_groundings = [agent for _, row in stmts_table.iterrows()
                  if row.nlm_true or row.nlm_false
                  for agent in row['agents']]

common_db_agents = Counter(db_groundings)
common_nlm_agents = Counter(nlm_groundings)

problematic = set(x[0] for x in common_db_agents.most_common(100))
problematic = problematic - set(nlm_groundings)

in_both = set(db_groundings) & set(nlm_groundings)

x = with_nlm[with_nlm['agents'].apply(lambda z: z <= in_both)]
x = x[x.pmid.apply(lambda z: z not in pmid_blacklist)]

sent_level = x.groupby('sentence_id').any()

z = extractions_df
z = z[(z.agent1 != '') & (z.agent2 != '')]

# keep only sentences that can be unambiguously
# matched to a sentence in as tokenized by REACH
z = z[z.sentence_id.isin(sent_level.index.values)]

# drop self interactions
z = z[z.apply(lambda row: len(row.agents) > 1
              and all(row.agents) and row.agent1 != row.agent2,
              axis=1)]

unreachable = sent_level[~sent_level.reach.astype('bool') &
                         sent_level.nlm_true.astype('bool')]
reach_and_nlm = sent_level[sent_level.reach.astype('bool') &
                           sent_level.nlm_true.astype('bool')]
reach_no_nlm = sent_level[sent_level.reach.astype('bool') &
                          ~sent_level.nlm_true.astype('bool')]

unreachable_df = z[z.sentence_id.isin(unreachable.index)]
unreachable_df = unreachable_df[unreachable_df.reader == 'nlm_ppi']

reach_no_nlm_df = z[z.sentence_id.isin(reach_no_nlm.index)]
reach_no_nlm_df = reach_no_nlm_df[reach_no_nlm_df.reader ==
                                  'reach']

reach_and_nlm_df = z[z.sentence_id.isin(reach_and_nlm.index)]
reach_nlm_groups = reach_and_nlm_df.groupby(['pmid', 'sentence', 'agents'])

new_rows = []
for (pmid, sentence, agents), group_df in reach_nlm_groups:
    new_row = {}
    print('***', len(group_df))
    for _, row in group_df.iterrows():
        if row.reader == 'nlm_ppi':
            new_row['agent1'] = row['agent1']
            new_row['agent2'] = row['agent2']
            new_row['nlm'] = row['stmt']
        elif row.reader == 'reach':
            new_row['reach'] = row['stmt']
    new_row['pmid'] = pmid
    new_row['sentence'] = sentence
    new_rows.append(new_row)

reach_and_nlm_df = pd.DataFrame(new_rows)
reach_and_nlm_df = reach_and_nlm_df[['pmid', 'agent1', 'agent2',
                                     'sentence', 'reach', 'nlm']]

unreachable_df['reach'] = 'None'
unreachable_df['nlm'] = unreachable_df['stmt']

reach_no_nlm_df['reach'] = reach_no_nlm_df['stmt']
reach_no_nlm_df['nlm'] = 'None'

final = pd.concat([unreachable_df, reach_no_nlm_df,
                   reach_and_nlm_df])
final = final[['pmid', 'agent1', 'agent2',
               'sentence', 'reach', 'nlm']]
final.to_csv('../result/extraction_spreadsheet.csv', sep=',',
             index=False)
