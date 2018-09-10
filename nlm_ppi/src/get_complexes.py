import pandas as pd
import re
from indra.sources import trips
from indra.statements import Agent, Complex, Evidence
from indra.literature.pmc_client import id_lookup
from indra.tools import assemble_corpus as ac
from functools import lru_cache


@lru_cache(10000)
def get_groundings(name):
    """Return a dictionary of groundings for a term.
    uses the trips processor.

    Parameters
    __________
    name : string
    text for a term we seek to ground

    Returns
    -------
    dict: dictionary of groundings
    """
    try:
        tp = trips.process_text(name)
        terms = tp.tree.findall('TERM')
        if not terms:
            return {}
        term_id = terms[0].attrib['id']
        agent = tp._get_agent_by_id(term_id, None)
        return agent.db_refs
    except Exception:
        return {}


@lru_cache(1000)
def get_pmid(pmcid):
    return id_lookup(pmcid, idtype='pmcid')['pmid']


class HGNC(object):
    def __init__(self):
        hgnc = pd.read_csv('~/indra/indra/resources/hgnc_entries.tsv',
                           sep='\t')
        hgnc['new_index'] = hgnc['HGNC ID'].apply(lambda x:
                                                  x.split(':')[1])
        hgnc = hgnc.set_index('new_index')
        self.__hgnc = hgnc
    
    def get_symbol(self, hgnc_id):
        return self.__hgnc.loc[hgnc_id]['Approved Symbol']


def get_agent(text):
    hgnc_converter = HGNC()
    db_refs = get_groundings(text)
    be = db_refs.get('FPLX')
    hgnc = db_refs.get('HGNC')
    if be:
        name = be
    elif hgnc:
        name = hgnc_converter.get_symbol(hgnc)
    else:
        name = re.sub(r"\s+", '_', text)
    return Agent(name=name, db_refs=db_refs)


def get_complex(row):
    agent1 = get_agent(row['arg1'])
    agent2 = get_agent(row['arg2'])
    pmid = get_pmid(row['pmc'])
    evidence = Evidence(source_api='nlm_ppi', source_id=row.index,
                        pmid=pmid, text=row['example'])
    return Complex(members=[agent1, agent2], evidence=evidence)


df = pd.read_excel("../input/pmcids_interactions1_full_predict1.xlsx",
                   dtype='str')
df['Statements'] = df.apply(get_complex, axis=1)

true_stmts = list(df[df['ppi'] == 'True']['Statements'].values)
false_stmts = list(df[df['ppi'] == 'False']['Statements'].values)

ac.dump_statements(true_stmts, '../work/nlm_ppi_true_statements.pkl')
ac.dump_statements(false_stmts, '../work/nlm_ppi_false_statements.pkl')
