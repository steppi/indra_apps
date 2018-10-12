import pandas as pd
import pickle
import networkx as nx
from itertools import combinations
from functools import lru_cache
from indra.databases.hgnc_client import get_hgnc_name

extr_df = pd.read_pickle('../work/extractions_table5.pkl')
with open('../work/all_reach_groundings.pkl', 'rb') as f:
    reach_ents = pickle.load(f)

fplx_map = pd.read_csv('~/indra/indra/resources/famplex_map.tsv',
                       sep='\t', names=['name_space', 'grounding', 'FPLX'])
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


@lru_cache(maxsize=10000)
def to_fplx(entry_id, name_space):
    fplx = fplx_map[fplx_map.name_space == name_space]
    fplx = fplx[fplx.grounding == entry_id]
    if len(fplx) > 0:
        return fplx.FPLX.values[0]
    else:
        return None


@lru_cache(maxsize=10000)
def to_hgnc(uniprot_id):
    hgnc = hgnc_map[hgnc_map.UniProt_ID == uniprot_id]
    if len(hgnc) > 0:
        hgnc_id = hgnc.HGNC_ID.values[0]
        hgnc_id = hgnc_id.split(':')[1]
        return hgnc_id
    else:
        return None


def agents_from_stmts(stmts):
    output = set([])
    for stmt in stmts:
        for agent in stmt.agent_list():
            db_refs = agent.db_refs.copy()
            if 'TEXT' in db_refs:
                db_refs['TEXT'] = db_refs['TEXT'].lower()
            output.add(frozenset(db_refs.items()))
    return list(output)


key_mapper = {'uniprot': 'UP',
              'be': 'FPLX',
              'interpro': 'IP',
              'pfam': 'PF',
              'uaz': 'uaz'}


def get_reach_json_groundings(reach_ents):
    grounding_set = set([])
    for sentence, groundings in reach_ents.items():
        for grounding in groundings:
            temp = set()
            temp.add(('TEXT', grounding[0].lower()))
            for key, value in grounding[1]:
                new_key = key_mapper[key]
                temp.add((new_key, value))
                if new_key == 'PF' or new_key == 'IP':
                    fplx = to_fplx(value, new_key)
                    if fplx:
                        temp.add(('FPLX', fplx))
                if new_key == 'UP':
                    hgnc = to_hgnc(value)
                    if hgnc:
                        temp.add(('HGNC', hgnc))
            grounding_set.add(frozenset(temp))
    return list(grounding_set)


def get_grounding_map():
    nlm_stmts = extr_df[extr_df['reader'] ==
                        'nlm_ppi']['stmt'].values

    db_stmts = extr_df[~(extr_df['reader'] ==
                         'nlm_ppi')]['stmt'].values

    nlm_groundings = agents_from_stmts(nlm_stmts)
    db_groundings = agents_from_stmts(db_stmts)
    reach_json_groundings = get_reach_json_groundings(reach_ents)

    mapper = {}
    G = nx.Graph()
    for grounding in nlm_groundings:
        G.add_node(grounding, source='nlm')
    for grounding in db_groundings:
        G.add_node(grounding, source='db')
    for grounding in reach_json_groundings:
        G.add_node(grounding, source='reach_json')
    node_pairs = combinations(G.nodes(), 2)
    for node1, node2 in node_pairs:
        if node1 & node2:
            G.add_edge(node1, node2)

    groundings_list = []
    for component in nx.connected_components(G):
        canonical = []
        sources = set([])
        for grounding in component:
            canonical.append(set(grounding))
            sources.add(G.node[grounding]['source'])
        canonical = set.union(*canonical)
        for grounding in component:
            mapper[grounding] = canonical
        groundings_list.append((canonical, sources))
    return mapper, groundings_list


problematic = []


def fix_grounding(row, mapper):
    """Use the above grounding map to fix groundings in the table frame.
    For use in a pd.apply
    """
    output = set([])
    for agent in row.stmt.agent_list():
        db_refs = agent.db_refs
        if 'TEXT' in db_refs:
            db_refs['TEXT'] = db_refs['TEXT'].lower()
        grounding = frozenset(db_refs.items())
        if grounding in mapper:
            output.add(frozenset(mapper[grounding]))
        else:
            problematic.append(grounding)
    return hash(frozenset(output))


# get the names of agents from agent keys. one off for use in pd.apply
# add agent names to extractions df
def __get_agent_names(row):
    return pd.Series(sorted(dict(agent)['TEXT'] if
                            'TEXT' in dict(agent) else
                            (get_hgnc_name(dict(agent)['HGNC'])
                            if 'HGNC' in dict(agent)
                             else '')
                            for agent in row.agents))


extr_df[['agent1',
         'agent2']] = extr_df.apply(__get_agent_names,
                                    axis=1)


mapper, groundings_list = get_grounding_map()
extr_df['agents'] = extr_df.apply(lambda x:
                                  fix_grounding(x, mapper),
                                  axis=1)

considered_agents = set([frozenset(x[0]) for x in groundings_list
                         if 'reach_json' in x[1] and
                         'nlm' in x[1]])

extr_df.to_csv('../work/extractions_table_agents_matched.tsv',
               sep='\t', index=False)
with open('../work/considered_agents.pkl', 'wb') as f:
    pickle.dump(considered_agents, f)
# frame = []
# for grounding in not_in_fplx:
#     text = grounding[0]
#     for name_space, entry in grounding[1]:
#         frame.append({'name_space': key_mapper[name_space],
#                       'text': text,
#                       'entry': entry})
# not_in_fplx_df = pd.DataFrame(frame)
# not_in_fplx_df = not_in_fplx_df[['name_space', 'entry', 'text']]
# not_in_fplx_df.sort_values(by=['name_space', 'entry'], inplace=True)
# not_in_fplx_df = not_in_fplx_df.groupby(['name_space',
#                                          'entry'],
#                                         as_index=False).agg(lambda x:
#                                                             ','.join(list(x)))

# not_in_fplx_df.to_csv('../result/fplx_missing_entries.txt', sep='\t',
#                       index=False)
