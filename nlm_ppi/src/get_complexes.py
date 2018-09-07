import pandas as pd
from indra.statements import Agent, Complex, Evidence
from indra.tools import assemble_corpus as ac
from indra.literature.pmc_client import id_lookup


def get_complex(row):
    agent1_dbrefs = {'HGNC': row['HGNC_arg1'], 'UP': row['UP_arg1']}
    agent2_dbrefs = {'HGNC': row['HGNC_arg2'], 'UP': row['HGNC_arg2']}
    agent1 = Agent(name=row['HGNC_arg1'], db_refs=agent1_dbrefs)
    agent2 = Agent(name=row['HGNC_arg2'], db_refs=agent2_dbrefs)
    ids = id_lookup(row['pmc'], idtype='pmcid')
    pmid = ids['pmid']
    evidence = Evidence(source_api='nlm_ppi', source_id=row.index,
                        pmid=pmid, text=row['example'])
    return Complex(members=[agent1, agent2], evidence=evidence)


df = pd.read_csv(("../work/pmcids_interactions1_full_predict1_"
                  "with_groundings.tsv"), sep='\t', dtype='str')
matrix = df.values
df['Statements'] = df.apply(get_complex, axis=1)

stmts = list(df['Statements'].values)

ac.dump_statements(stmts, "../work/nlm_ppi_statements.pkl")
