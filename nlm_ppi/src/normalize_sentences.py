from indra.tools import assemble_corpus as ac
import pandas as pd
import networkx as nx
from copy import deepcopy
import pickle
from fuzzywuzzy import fuzz
from itertools import combinations
from indra.statements import Complex, Evidence


def filter_none(stmts):
    """Filter statements including an empty agent.

    Creates a copy. Does not work in place

    Parameters
    __________
    stmts : list of Indra Statements

    Returns
    -------
    list of Indra Statements : subset of input statements that contain two or
    more agents.
    """
    filtered = []
    for stmt in stmts:
        if all(stmt.agent_list()) and len(stmt.agent_list()) > 1:
            filtered.append(deepcopy(stmt))
    return filtered


def split_complexes(stmts):
    """Split complexe statements  with more than two members into multiple
    complex statements.

    Parameters
    ----------
    stmts : list of Indra Statements

    Returns
    list of Indra Statements : a potentially larger list of indra statements.
    each complex statement with more than two members will be replaced with
    multiple complex statements: one statement for each pair of agents in the
    original complex.
    """
    filtered = []
    for stmt in stmts:
        if len(stmt.agent_list()) > 2 and type(stmt) == Complex:
            text = stmt.evidence[0].text
            pmid = stmt.evidence[0].pmid
            source_api = stmt.evidence[0].source_api
            source_id = stmt.evidence[0].source_id
            evidence = Evidence(source_api=source_api, source_id=source_id,
                                pmid=pmid, text=text)
            for agent1, agent2 in combinations(stmt.agent_list(), 2):
                new_stmt = Complex(members=[agent1, agent2],
                                   evidence=evidence)
                filtered.append(new_stmt)
        else:
            filtered.append(deepcopy(stmt))
    return filtered


def sentence_match(sen1, sen2, cutoff1=0.8, cutoff2=0.5):
    """Determine if sentences preprocessed by different readers are actually
    the same.

    Parameters
    ----------
    sen1: string
    sen2: string
    Two strings containing sentences

    cutoff1: float
    two sentences match if the levenstein ratio between them is greater than
    cutoff1

    cutoff2: float
    if the levenstein ratio is less than cutoff1 but greater than cutoff2, the
    sentences match if the partial_ratio between sentence1 and sentence2 is
    greater than cutoff1

    Returns
    -------
    bool: True if the sentences match
    """
    ratio = fuzz.ratio(sen1, sen2)/100
    if ratio > cutoff1:
        return True
    elif ratio > cutoff2:
        return fuzz.partial_ratio(sen1, sen2)
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
    groups = new_mapping_df.groupby(level=['pmid'])
    for _, new_df in groups:
        G = nx.Graph()
        for index, extraction in new_df.iterrows():
            G.add_node(index, sentence=extraction['normed_sentence'])
        node_pairs = combinations(G.nodes(data=True), 2)
        for (index1, data1), (index2, data2) in node_pairs:
            sentence1, sentence2 = (data1['sentence'],
                                    data2['sentence'])
            if sentence_match(sentence1, sentence2,
                              cutoff1=cutoff1, cutoff2=cutoff2):
                G.add_edge(index1, index2)
        for component in nx.connected_components(G):
            least_sentence = min(G.node[x]['sentence']
                                 for x in component)
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
