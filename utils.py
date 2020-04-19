import numpy as np
import pandas as pd

DATA_LEN = 46972
VOCAB_SIZE = 29908


def read_word_dict(model_dir):
    word_dict = {}
    word_dict[None] = str(len(word_dict))
    with open(model_dir / "vocab.all", "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            word = line.strip()
            word_dict[word] = str(len(word_dict))
    return word_dict


def get_ans():
    file_path = "../data/queries/ans_train.csv"
    df = pd.read_csv(file_path)
    docs_lst = df["retrieved_docs"].values.tolist()
    ans_lst = [docs.split() for docs in docs_lst]
    return ans_lst


def load_pkl(pkl_path):
    import pickle
    with open(pkl_path, mode='rb') as f:
        obj = pickle.load(f)

    return obj


def get_file_lst():
    file_lst = []
    with open("../data/model/file-list", "r", encoding="utf-8") as f:
        for line in f:
            file_lst.append(line.strip())
    return file_lst


def idx2file_name(idx_lst):
    file_lst = get_file_lst()
    file_name = [name.split('/')[-1] for name in file_lst]
    ans_name = [file_name[idx] for idx in idx_lst]
    return ans_name


def get_rankn_bigram(rank_lst, docs):
    return [docs[idx]['bigram'] for idx in rank_lst]


def SubmitGenerator(prediction, filename='prediction.csv'):
    """
    Args:
        prediction (numpy array)
        filename (str)
    """
    submit = {}
    submit['query_id'] = []
    submit['retrieved_docs'] = []
    for idx, p in enumerate(prediction):
        submit['query_id'].append(f"{(idx+11):03}")
        submit['retrieved_docs'].append(" ".join(p).lower())
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)
