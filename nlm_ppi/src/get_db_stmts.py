import pickle
from indra.tools import assemble_corpus as ac
from indra.sources import indra_db_rest as idr
from indra.literature import pmc_client
from indra.db import get_primary_db
from indra.db.util import distill_stmts

db = get_primary_db()

with open('../pmcids_interactions.txt', 'rt') as f:
    pmcids = [line.strip() for line in f.readlines()]

stmts_by_pmid = {}
for pmcid in pmcids:
    ids = pmc_client.id_lookup(pmcid, 'pmcid')
    assert 'pmid' in ids
    pmid = ids['pmid']
    clauses = [db.TextRef.pmid.in_([pmid])] + \
                             db.link(db.TextRef, db.RawStatements)
    stmts = distill_stmts(db, get_full_stmts=True, clauses=clauses)
    stmts_by_pmid[pmid] = list(stmts)

ac.dump_statements(stmts_by_pmid, 'db_interaction_stmts_by_pmid.pkl')

