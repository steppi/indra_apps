from indra.tools import assemble_corpus as ac
import pandas as pd
import numpy as np
import networkx as nx
from copy import deepcopy
import pickle
from fuzzywuzzy import fuzz
from itertools import combinations
from indra.statements import Complex, Evidence
from cachetools import cached, LRUCache
from cachetools.keys import hashkey


@cached(LRUCache(maxsize=500), key=lambda x:
        hashkey((x.sentence,
                 x.reach_sentences)))
def sentence_id(x,
                cutoff=0.85):
    sentence = x.sentence
    sentence_set = x.reach_sentences
    fuzz_values = np.fromiter((fuzz.ratio(sentence,
                                          sentence2)/100
                               for _, sentence2 in
                               sentence_set),
                              dtype=float)
    match = np.where(fuzz_values > cutoff)[0]
    if len(match) == 1:
        return sentence_set[match[0]][0]
    else:
        return None


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
    """Split complex statements with more than two members into multiple
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


def sentence_match(sen1, sen2, cutoff=0.8):
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
    return ratio > cutoff


def get_pmid_mapping(stmts):
    """Build a dataframe containing all info from list of statements

    Parameters
    __________

    stmts : list(tuple)
    list of tuples of the form (indra.statements.Statement, bool)
    the bool expresses if the statement has been predicted as true
    or false_stmts

    Returns
    _______

    pandas.DataFrame : dataframe with rows corresponding to extractions
    and columns:
    pmid, agents, reader, type, sentence, stmt
    [
    pmid : string
    PubMed ID of the article the extraction was taken from

    agents: frozenset of frozensets of tuples
    identifies the two agents in corresponding statement disregarding order

    reader : string
    reader that made the extraction

    type : bool
    describes if statement was predicted as true or false

    sentence : string
    sentence text where statement was extracted

    stmt : indra.statements.Statement
    indra statement corresponding to extraction
    ]

    keeps only one extraction in each sentence that has the same two agents
    for the same reader
    """
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


def match_sentences(mapping_df, cutoff=0.8):
    """matches sentences from different readers which cannot be identified
    directly because they have been preprocessed differently

    Makes use of fuzzy string matching
    Parameters
    ----------

    mapping_df : pandas.DataFrame
    dataframe of extraction information generated with get_pmid_mapping

    cutoff1 : float
    two sentences are considered identical if their levenstein ratio is greater
    than cutoff1

    cutoff2 : float
    two sentences are also considered identical if their levenstein ratio is
    greater than cutoff2 and their partial levenstein ratio is greater than
    cutoff1.

    Returns
    -------

    pandas.DataFrame : copy of input dataframe with a new column, sentence ID.
    sentence ID is identical for sentences that have been matched with fuzzy
    string matching
    """
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
            if sentence_match(sentence1, sentence2, cutoff=cutoff):
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


""" Reads in nlm statements and db statements. Filters out statements with
a Nonetype agent (e.g. B is phosphorylated vs A phosphorylates B).
Splits complex statements with more than two agents into multiple complex
statements with two agents each. Builds dataframe of extraction
information. Matches identical sentences processed differently by different
readers and dumps output into a pickle file.
"""

pmid_blacklist = set(['24141421', '28855251', '25822970'])
nlm_true = ac.load_statements('../work/nlm_ppi_true_statements.pkl')
nlm_false = ac.load_statements('../work/nlm_ppi_false_statements.pkl')
with open('../work/db_interaction_stmts_by_pmid.pkl', 'rb') as f:
    db_pmid_mapping = pickle.load(f)
with open('../work/all_reach_sentences.pkl', 'rb') as f:
    reach_sentences = pickle.load(f)

db_stmts = []
for pmid, extraction in db_pmid_mapping.items():
    for stmt in extraction:
        db_stmts.append(stmt)
nlm_true = filter_none(nlm_true)
nlm_false = filter_none(nlm_false)
db_stmts = filter_none(db_stmts)

nlm_true = split_complexes(nlm_true)
nlm_false = split_complexes(nlm_false)
db_stmts = split_complexes(db_stmts)

nlm_stmts = [(stmt, True) for stmt in nlm_true]
nlm_stmts += [(stmt, False) for stmt in nlm_false]
db_stmts = [(stmt, True) for stmt in db_stmts]
stmts = nlm_stmts + db_stmts

mapping_df = get_pmid_mapping(stmts)
# keep only reach and nlm statements for now
mapping_df = mapping_df[~mapping_df.isin(pmid_blacklist)].dropna()
mapping_df['reach_sentences'] = mapping_df.pmid.apply(lambda x:
                                                      reach_sentences.get(x))
mapping_df['sentence_id'] = mapping_df.apply(lambda x:
                                             sentence_id(x), axis=1)
mapping_df = mapping_df.dropna()
# dedup = match_sentences(mapping_df, cutoff=0.85)
mapping_df.to_pickle('../work/extractions_table4.pkl')
