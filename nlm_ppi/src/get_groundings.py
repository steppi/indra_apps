import pandas as pd
from indra.sources import trips
from functools import lru_cache


@lru_cache(1000)
def get_groundings(name):
    try:
        tp = trips.process_text(name)
        terms = tp.tree.findall('TERM')
        if not terms:
            return None, None
        term_id = terms[0].attrib['id']
        agent = tp._get_agent_by_id(term_id, None)
        return agent.db_refs.get('HGNC'), agent.db_refs.get('UP')
    except Exception:
        return None, None


df = pd.read_excel("../input/pmcids_interactions1_full_predict1.xlsx")
df['HGNC_arg1'], df['UP_arg1'] = zip(*df['arg1'].map(get_groundings))
df['HGNC_arg2'], df['UP_arg2'] = zip(*df['arg2'].map(get_groundings))

df.to_csv(("../work/pmcids_interactions1_full_predict1_"
           "with_groundings.tsv"), sep="\t", index=False)
