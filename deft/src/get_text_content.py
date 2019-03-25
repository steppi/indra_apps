import pickle

from indra_db.util.content_scripts import get_text_content_from_stmt_ids


if __name__ == '__main__':
    with open('/deft_drive/ROS_stmts.pkl', 'rb') as f:
        all_stmts = pickle.load(f)

    for shortform, stmts in all_stmts.items():
        if len(stmts) > 5000:
            print(shortform)
            text_dict = get_text_content_from_stmt_ids(stmts)
            with open(f'/deft_drive/{shortform}_texts.pkl', 'wb') as f:
                pickle.dump(text_dict, f)
