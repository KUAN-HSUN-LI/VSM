import pdb
import numpy as np
import multiprocessing
import pickle
import pandas as pd
from utils import load_pkl
from cli import get_args
from pathlib import Path
import os
DATA_LEN = 46972


def get_inverted_data(model_dir):
    with open(model_dir / "inverted-file", "r") as f:
        unigram_idf = {}
        bigram_idf = {}
        doc_datas = [{'doc_len': 0, 'unigram': {}, 'bigram': {}} for _ in range(DATA_LEN)]
        while True:
            head_line = f.readline().strip()
            if head_line == "":
                break
            head_line = list(map(int, head_line.split()))
            head_idx = head_line[0]
            print(head_idx, end='\r')
            if head_line[1] == -1:
                unigram_idf[str(head_idx)] = np.log(DATA_LEN / head_line[2])
            else:
                bigram_idf[str(head_idx) + " " + str(head_line[1])] = np.log(DATA_LEN / head_line[2])
            for _ in range(head_line[2]):
                line = f.readline()
                line = list(map(int, line.strip().split()))
                if head_line[1] == -1:
                    doc_datas[line[0]]['doc_len'] += line[1]
                    doc_datas[line[0]]['unigram'][str(head_idx)] = line[1]
                else:
                    doc_datas[line[0]]['bigram'][str(head_line[0]) + " " + str(head_line[1])] = line[1]
    return unigram_idf, bigram_idf, doc_datas


if __name__ == "__main__":
    args = get_args()
    if os.path.exists("unigram_idf.pkl") and os.path.exists("bigram_idf.pkl") and os.path.exists("doc_datas.pkl"):
        pass
    else:
        unigram_idf, bigram_idf, doc_datas = get_inverted_data(args.model_dir)
        with open("unigram_idf.pkl", "wb") as f:
            pickle.dump(unigram_idf, f)

        with open("bigram_idf.pkl", "wb") as f:
            pickle.dump(bigram_idf, f)

        with open("doc_datas.pkl", "wb") as f:
            pickle.dump(doc_datas, f)
