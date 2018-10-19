import pandas as pd
import argparse
from indra.tools import assemble_corpus as ac
from indra.databases.acromine_client import get_disambiguations
from random import sample


def _get_longforms(acro, shortform):
    """ Get the longforms in an acromine json associated to a particular
    shortform
    for use in pandas.apply

    Parameters
    ---------
    acro: list[dict] pyobject representation of acromine json
    shortform: str shortform text for an agent
    Returns
    -------
    longforms: list[string]
    list of associated longforms from all elements of acromine json with
    the given shortform
    """
    longforms = [disamb.get('longForm').lower() for disamb in acro
                 if disamb.get('shortForm') == shortform]
    return longforms


def _get_agent_names(stmt, text):
    """ Get the agent names for all agents in a stmt with a particular TEXT ref

    Paramters
    ---------
    stmt: indra.statement
    text: str an agent text

    Returns
    -------
    names: list[str] list of agent names for agents in stmt with given text
    """
    if stmt:
        names = [agent.name for agent in stmt.agent_list()
                 if agent and agent.db_refs.get('TEXT') == text]
    else:
        names = []
    return names


def get_stmts_df(stmts, shortform):
    """ Given a list of statements and a particular shortform, run the
    acromine disambiguation tool on the stmt texts and return a dataframe with
    disambiguation information for a particular shortform.

    Parameters
    ----------
    stmts: list[indra.statement] list of indra statements. should contain
    statements with at least one agent text with a particular shortform
    shortform: str shortform text

    Returns
    -------
    stmts_df: pandas.DataFrame dataframe containing acromine disambiguation
    information
    """
    # Place statementss into a dataframe
    stmts_df = pd.DataFrame(stmts, columns=['stmt'])
    # Add column for the entity text
    stmts_df['text'] = stmts_df.stmt.apply(lambda x: x.evidence[0].text)
    # Drop statements with missing text
    stmts_df = stmts_df.dropna()
    # Add column with acromine jsons associated to stmt texts feeding them to
    # the REST API
    stmts_df['acromine'] = stmts_df.text.apply(get_disambiguations)
    # Add column of longform texts associated to agents with the shortform
    # given as input to this function
    stmts_df['longform'] = stmts_df.acromine.apply(lambda x:
                                                   _get_longforms(x,
                                                                  shortform))
    # Filter out statements that contain either no agents with this particular
    # shortform or statements where acromine has disambiguated the same text
    # multiple ways in the same sentence
    stmts_df = stmts_df[stmts_df.longform.apply(lambda x: len(set(x)) == 1)]
    # Replace list of longforms with text of the first element, having ensured
    # that longforms has only one element
    stmts_df.loc[:, 'longform'] = stmts_df.longform.apply(lambda x: x[0])
    # Get agent names for all agents in stmts where the input shortform is the
    # agent text
    stmts_df['grounding'] = stmts_df.stmt.apply(lambda x:
                                                _get_agent_names(x,
                                                                 shortform))

    # As before, filter if there is not one unique grounding and replace list
    # with the unique agent name
    stmts_df = stmts_df[stmts_df.grounding.apply(lambda x: len(set(x)) == 1)]
    stmts_df.loc[:, 'grounding'] = stmts_df.grounding.apply(lambda x: x[0])
    return stmts_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run acromine on the texts'
                                     'from a list of statements using the REST'
                                     ' API. Output spreadsheet of results')
    parser.add_argument('shortform')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('-c', '--count', help='number of statements to run'
                        ' acromine on. take random subset to avoid hammering'
                        'the REST API',
                        type=int,
                        default=500)

    results = parser.parse_args()

    shortform = results.shortform
    infile = results.infile
    outfile = results.outfile
    count = results.count

    stmts = ac.load_statements(infile)
    subset_stmts = sample(stmts, count)
    
    subset_df = get_stmts_df(subset_stmts, shortform)
    subset_df.to_csv(outfile, sep=',', index=False)
