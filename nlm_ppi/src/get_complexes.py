import pandas as pd
import re
from indra.sources import trips
from indra.statements import Agent, Complex, Evidence
from indra.literature.pmc_client import id_lookup
from indra.tools import assemble_corpus as ac
from functools import lru_cache
from indra.databases.hgnc_client import get_hgnc_name


@lru_cache(1000)
def get_dbrefs(name):
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
        term_id = terms[0].attrib['id']
        agent = tp._get_agent_by_id(term_id, None)
        db_refs = agent.db_refs
    except Exception:
        db_refs = None
    finally:
        if not db_refs:
            db_refs = ac.grounding_map.get(name.upper())
        return db_refs


@lru_cache(1000)
def get_pmid(pmcid):
    return id_lookup(pmcid, idtype='pmcid')['pmid']


_hgnc = pd.read_csv('~/indra/indra/resources/hgnc_entries.tsv',
                    sep='\t')
_hgnc['new_index'] = _hgnc['HGNC ID'].apply(lambda x:
                                            x.split(':')[1])
_hgnc = _hgnc.set_index('new_index')


def get_agent(text):
    db_refs = get_dbrefs(text)
    # check the grounding map if trips couldn't find a grounding
    be = db_refs.get('FPLX')
    hgnc_id = db_refs.get('HGNC')
    if be:
        name = be
    elif hgnc_id:
        name = get_hgnc_name(hgnc_id)
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


if __name__ == '__main__':
    df = pd.read_excel("../input/pmcids_interactions1_full_predict1.xlsx",
                       dtype='str')
    df['Statements'] = df.apply(get_complex, axis=1)

    true_stmts = list(df[df['ppi'] == 'True']['Statements'].values)
    false_stmts = list(df[df['ppi'] == 'False']['Statements'].values)

    ac.dump_statements(true_stmts, '../work/nlm_ppi_true_statements.pkl')
    ac.dump_statements(false_stmts, '../work/nlm_ppi_false_statements.pkl')