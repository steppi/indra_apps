import pandas as pd
from copy import deepcopy
from cachetools import cached, LRUCache
import numpy as np
from statsmodels.stats.proportion import proportion_confint as confint


# many apologies to anyone who has to use this or modify it, including
# future albert.

@cached(LRUCache(maxsize=500), key=lambda x:
        tuple(sorted([agent.name for agent
                      in x.agent_list()])))
def translation_map(stmt):
    """This is for matching groundings manually."""
    mapper = {'polbeta': 'dna polymerase beta',
              'tfn': 'transferrin',
              'rxralpha': 'rxra',
              'ran': 'ara24',
              'aid': 'aicda',
              'signal transducer and_activator of transcription (stat)-3':
              'stat3',
              'amyloid precursor protein': 'app',
              'protein interacting with c kinase 1': 'pick1',
              'glyceraldehyde 3-phosphatase dehydrogenase': 'gapdh',
              'upar': 'urokinase-type plasminogen activator'}
    new_stmt = deepcopy(stmt)
    for agent in new_stmt.agent_list():
        if agent.name in mapper:
            new_ground = get_dbrefs(mapper[agent.name])
            agent.db_refs = new_ground
            agent.db_refs['TEXT'] = agent.name
    return new_stmt


def get_statements_df(extractions_df):
    """Alternative format for extractions
    the columns are
    pmid sentence_id sentence agents sparser reach nlm_true nlm_false
    where the columns corresponding to the readers contain actual the
    actual indra statements that were extracted"""
    groups = extractions_df.groupby(by=['pmid', 'agents', 'sentence_id'],
                                    as_index=False)
    frame = []
    for index1, new_df, in groups:
        new_row = {'reach': '', 'sparser': '',
                   'nlm_true': '', 'nlm_false': ''}
        for index2, row in new_df.iterrows():
            if row.reader == 'nlm_ppi':
                if row['type']:
                    new_row['nlm_true'] = row.stmt
                else:
                    new_row['nlm_false'] = row.stmt
            else:
                new_row[row.reader] = row.stmt
            new_row['pmid'] = row['pmid']
            new_row['sentence'] = row['sentence']
            new_row['agents'] = row['agents']
            new_row['sentence_id'] = row['sentence_id']
            frame.append(new_row)
    output = pd.DataFrame(frame)
    output = output[['pmid', 'sentence_id', 'sentence', 'agents',
                     'sparser', 'reach', 'nlm_true', 'nlm_false']]
    return output


# read in extractions dataframe
extr_df = pd.read_pickle('../work/extractions_table_agents_matched.pkl')
# read in set of agents that are recognizable by both nlm and reach
all_agents = pd.read_pickle('../work/groundings_table.pkl')


# in this next step we remove sentences that contain large numbers of genes,
# such as tables and gene lists. this removes sentences with more than 60
# extractions by the various readers. magic number chosen based on manual
# inspection
sent_counts = extr_df.sentence_id.value_counts()
extr_df = extr_df[extr_df.sentence_id.apply(lambda x:
                                            sent_counts.loc[x] < 60)]


def nlm_and_reach(agents_key):
    """tests if all of the agents from an extraction are recognizable
    by both reach and nlm.
    """
    # agents_key is a frozenset of two agent hashes
    # unpack into two hashes
    if len(agents_key) == 2:
        hash1, hash2 = tuple(agents_key)
    else:
        hash1 = tuple(agents_key)[0]
        hash2 = hash1
    agent1, agent2 = all_agents.loc[hash1], all_agents.loc[hash2]
    cond_agent1 = agent1.nlm and agent1.reach_json
    cond_agent2 = agent2.nlm and agent2.reach_json
    return cond_agent1 and cond_agent2


# only include extractions where all agents can be recognized by
# both nlm and reach. (ignore sparser for now)
extr_df = extr_df[extr_df.agents.apply(nlm_and_reach)]


# convert extractions to alternative format. see doc_string for
# get_statements_df
stmts_table = get_statements_df(extr_df)
stmts_table = stmts_table.sort_values(['pmid', 'sentence_id'])

# this section is for identifying agents that are recognized by both
# reach and nlm. sorry to anyone who has to work with this. including
# future albert


with_nlm = stmts_table[stmts_table.nlm_true.astype('bool') |
                       stmts_table.nlm_false.astype('bool')]
sent_level = with_nlm.groupby('sentence_id').any()

z = extr_df
z = z[(z.agent1 != '') & (z.agent2 != '')]


# keep only sentences that can be unambiguously
# matched to a sentence as tokenized by REACH
z = z[z.sentence_id.isin(sent_level.index.values)]

# drop self interactions
z = z[z.apply(lambda row: len(row.agents) > 1
              and all(row.agents) and row.agent1 != row.agent2,
              axis=1)]

z['foundby'] = z.apply(lambda x:
                       x.stmt.evidence[0].annotations.get('found_by')
                       if x.reader == 'reach' else None, axis=1)

unreachable = sent_level[~sent_level.reach.astype('bool') &
                         sent_level.nlm_true.astype('bool')]
reach_and_nlm = sent_level[sent_level.reach.astype('bool') &
                           sent_level.nlm_true.astype('bool')]
reach_no_nlm = sent_level[sent_level.reach.astype('bool') &
                          ~sent_level.nlm_true.astype('bool')]
no_reach_no_nlm = sent_level[~sent_level.reach.astype('bool') &
                             ~sent_level.nlm_true.astype('bool')]


unreachable_df = z[z.sentence_id.isin(unreachable.index)]
unreachable_df = unreachable_df[unreachable_df.reader == 'nlm_ppi']

reach_no_nlm_df = z[z.sentence_id.isin(reach_no_nlm.index)]
reach_no_nlm_df = reach_no_nlm_df[reach_no_nlm_df.reader ==
                                  'reach']

reach_and_nlm_df = z[z.sentence_id.isin(reach_and_nlm.index)]
reach_nlm_groups = reach_and_nlm_df.groupby(['pmid', 'sentence', 'agents'])

unreachable_df['reach'] = None
unreachable_df['nlm'] = unreachable_df['stmt']

reach_no_nlm_df['reach'] = reach_no_nlm_df['stmt']
reach_no_nlm_df['nlm'] = None

final = pd.concat([unreachable_df, reach_no_nlm_df])
final = final[['pmid', 'agent1', 'agent2',
               'sentence', 'reach', 'nlm', 'foundby']]
# final.to_csv('../result/extraction_spreadsheet_revised_2.csv', sep=',',
#              index=False)

# finding rules that nlm is likely to miss
reach_no_nlm_rules = reach_no_nlm_df['foundby'].value_counts()
reach_and_nlm_rules = reach_and_nlm_df['foundby'].value_counts()


reach_with_nlm = reach_and_nlm_df[reach_and_nlm_df.reader == 'reach']
reach_no_nlm = reach_no_nlm_df[reach_no_nlm_df.reader == 'reach']
reach_with_nlm['nlm'] = True
reach_no_nlm['nlm'] = False
reach_extractions = pd.concat([reach_with_nlm, reach_no_nlm],
                              sort=True)

reach_extractions = reach_extractions.drop(['reach', 'reach_sentences'],
                                           axis=1)
reach_groups = reach_extractions.groupby(['foundby', 'nlm'])
reach_counts = pd.Series(reach_groups.count()['agents'])
reach_counts = reach_counts.to_frame(name='count')
reach_counts = reach_counts.reset_index()

with_nlm = reach_counts[reach_counts.nlm]
no_nlm = reach_counts[~reach_counts.nlm]

with_nlm.drop('nlm', inplace=True, axis=1)
no_nlm.drop('nlm', inplace=True, axis=1)
with_nlm.columns = ['foundby', 'with_nlm']
no_nlm.columns = ['foundby', 'no_nlm']

counts = pd.merge(with_nlm, no_nlm, how='outer', on='foundby')
counts.fillna(0, inplace=True)
counts[['with_nlm', 'no_nlm']] = counts[['with_nlm',
                                         'no_nlm']].astype(int, inplace=True)
counts['total'] = counts.with_nlm + counts.no_nlm
counts['prob'] = counts.with_nlm / counts.total
counts['sigma'] = counts.prob*(1-counts.prob)/counts.total
counts['sigma'] = counts.sigma.apply(np.sqrt)
counts[['lower', 'upper']] = counts.apply(lambda x:
                                          pd.Series(confint(x.with_nlm,
                                                            x.total,
                                                            alpha=0.0025)),
                                          axis=1)
counts.set_index('foundby', inplace=True)

final['nlm_sents_with_rule'] = final.foundby.apply(lambda x:
                                                   counts.loc[x].with_nlm
                                                   if x
                                                   else None)

final['total_sents_with_rule'] = final.foundby.apply(lambda x:
                                                     counts.loc[x].total
                                                     if x
                                                     else None)

final['prob'] = final.foundby.apply(lambda x: counts.loc[x].prob if x
                                    else None)

final['lower'] = final.foundby.apply(lambda x: counts.loc[x].lower if x
                                     and counts.loc[x].total >= 10
                                     else None)
final['upper'] = final.foundby.apply(lambda x: counts.loc[x].upper
                                     if x and counts.loc[x].total >= 10
                                     else None)

# final.to_csv('../result/extraction_spreadsheet_with_reach_rules.csv', sep=',',
#              index=False)


w = x[x.reach.astype('bool') &
      x.nlm_true.astype('bool')]
w['foundby'] = w.reach.apply(lambda x:
                             x.evidence[0].annotations.get('found_by'))
w['nlm_sents_with_rule'] = w.foundby.apply(lambda x:
                                           counts.loc[x].with_nlm
                                           if x
                                           else None)

w['total_sents_with_rule'] = w.foundby.apply(lambda x:
                                             counts.loc[x].total
                                             if x
                                             else None)

w['prob'] = w.foundby.apply(lambda x: counts.loc[x].prob if x else None)

w['lower'] = w.foundby.apply(lambda x: counts.loc[x].lower if x
                             and counts.loc[x].total >= 10
                             else None)
w['upper'] = w.foundby.apply(lambda x: counts.loc[x].upper
                             if x and counts.loc[x].total >= 10
                             else None)
w[['agent1', 'agent2']] = w.apply(__get_agent_names,
                                  axis=1)

w = w.groupby(['agents', 'sentence'],
              as_index=False).first()

# w[['pmid', 'agent1', 'agent2', 'sentence',
#    'reach', 'nlm_true', 'foundby',
#    'nlm_sents_with_rule',
#    'total_sents_with_rule',
#    'prob', 'lower', 'upper']].to_csv('../result/nlm_reach_extractions.csv',
#                                      sep=',',
#                                      index=False)
