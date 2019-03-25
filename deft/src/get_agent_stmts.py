import pickle

from indra_db.util.content_scripts import get_stmts_with_agent_text_like

if __name__ == '__main__':
    two_letter = get_stmts_with_agent_text_like('ROS',
                                                filter_genes=True)
    with open('/deft_drive/ROS_stmts.pkl', 'wb') as f:
        pickle.dump(two_letter, f)
