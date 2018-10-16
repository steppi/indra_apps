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
from nltk.tokenize import sent_tokenize


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


@cached(LRUCache(maxsize=500), key=lambda x:
        hashkey((x.sentence,
                 x.reach_sentences)))
def split_differently(x, cutoff=0.85):
    def max_ratio(tokens1, tokens2):
        fuzz_values = np.fromiter((fuzz.ratio(token1,
                                              token2)/100
                                   for token1 in tokens1
                                   for token2 in tokens2),
                                  dtype=float)
        return np.max(fuzz_values)
    sentence = x.sentence
    sent1 = sent_tokenize(sentence)
    reach_sentences = x.reach_sentences
    reach_tokenized = [sent_tokenize(sent)
                       for _, sent in reach_sentences]
    fuzz_values = np.fromiter((max_ratio(sent1,
                                         sent2)
                               for sent2 in reach_tokenized),
                              dtype=float)
    match = np.where(fuzz_values > cutoff)[0]
    if len(match) == 1:
        r = fuzz.ratio(sentence,
                       reach_sentences[match[0]][1])/100
        if r < 0.7 and r > 0.5:
            return (sentence, reach_sentences[match[0]][1])
        else:
            return None
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


""" Reads in nlm statements and db statements. Filters out statements with
a Nonetype agent (e.g. B is phosphorylated vs A phosphorylates B).
Splits complex statements with more than two agents into multiple complex
statements with two agents each. Builds dataframe of extraction
information. Matches identical sentences processed differently by different
readers and dumps output into a pickle file.
"""

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
mapping_df['reach_sentences'] = mapping_df.pmid.apply(lambda x:
                                                      reach_sentences.get(x))
mapping_df['sentence_id'] = mapping_df.apply(lambda x:
                                             sentence_id(x), axis=1)
mapping_df = mapping_df.dropna()
# find sentence_ids that contain nlm statements
nlm_ids = mapping_df['sentence_id'][mapping_df.reader == 'nlm_ppi'].unique()
mapping_df = mapping_df[mapping_df['sentence_id'].isin(nlm_ids)]

mapping_df.to_pickle('../work/extractions_table5.pkl')
