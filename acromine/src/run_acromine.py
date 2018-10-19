import pandas as pd
from indra.tools import assemble_corpus as ac
from indra.databases.acromine_client import get_disambiguations
from random import sample


def get_longforms(acro, shortform):
    longforms = [disamb['longForm'].lower() for disamb in acro
                 if disamb['shortForm'] == shortform]
    return longforms


def get_groundings(stmt, text):
    if stmt:
        print(stmt)
        names = [agent.name for agent in stmt.agent_list()
                 if agent and agent.db_refs.get('TEXT') == text]
    else:
        names = []
    return names


def get_stmts_df(stmts, shortform):
    stmts_df = pd.DataFrame(stmts, columns=['stmt'])
    stmts_df['text'] = stmts_df.stmt.apply(lambda x: x.evidence[0].text)
    stmts_df = stmts_df.dropna()
    stmts_df['acromine'] = stmts_df.text.apply(get_disambiguations)
    stmts_df['longform'] = stmts_df.acromine.apply(lambda x:
                                                   get_longforms(x, shortform))
    stmts_df = stmts_df[stmts_df.longform.apply(lambda x: len(set(x)) == 1)]
    stmts_df.loc[:, 'longform'] = stmts_df.longform.apply(lambda x: x[0])
    stmts_df['grounding'] = stmts_df.stmt.apply(lambda x:
                                                get_groundings(x, shortform))
    stmts_df = stmts_df[stmts_df.grounding.apply(lambda x: len(set(x)) == 1)]
    stmts_df.loc[:, 'grounding'] = stmts_df.grounding.apply(lambda x: x[0])
    return stmts_df


if __name__ == '__main__':
    stmts = ac.load_statements('../work/ER_statements.pkl')
    test_stmts = sample(stmts, 10)
    test_df = get_stmts_df(test_stmts, 'ER')
    test_df.to_pickle('../work/sample_ER_statements_df_old.pkl')
