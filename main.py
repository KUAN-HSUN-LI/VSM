import pdb
from utils import read_word_dict, load_pkl, idx2file_name, SubmitGenerator, get_rankn_bigram
from cli import get_args
import multiprocessing as mp
from query_process import get_query
from statistics import mean
from feedback import rocchio_feedback
k1 = 2.0
b = 0.75
k2 = 500


args = get_args()
print("processing doc")
word_dict = read_word_dict(args.model_dir)
unigram_idf = load_pkl("unigram_idf.pkl")
bigram_idf = load_pkl("bigram_idf.pkl")
doc_datas = load_pkl("doc_datas.pkl")
print("processing query")
query_data = get_query(args.query_file, word_dict)
doc_ave = mean(d['doc_len'] for d in doc_datas)
for idx, doc in enumerate(doc_datas):
    print("idx", idx, end='\r')
    for key in doc['bigram'].keys():
        doc['bigram'][key] = (k1+1) * doc['bigram'][key] / (doc['bigram'][key] + k1 * (1 - b + b * (doc['doc_len'] / doc_ave))) * bigram_idf[key]


def calc_relation(doc, query):
    score = 0
    for key, value in doc.items():
        if key in query:
            score += value * (k2+1) * query[key] / (query[key] + k2)
    return score


def test(query):
    relations = []
    for doc in doc_datas:
        relations.append(calc_relation(doc['bigram'], query))
    relations_idx = sorted(range(len(relations)), key=lambda k: relations[k])
    rankn = idx2file_name(relations_idx[-1:-101:-1])
    return rankn


def with_feedback(query):
    relations = []
    for doc in doc_datas:
        relations.append(calc_relation(doc['bigram'], query))
    relations_idx = sorted(range(len(relations)), key=lambda k: relations[k])

    query = rocchio_feedback(query, get_rankn_bigram(relations_idx[-1:-101:-1], doc_datas), 10)
    relations = []
    for doc in doc_datas:
        relations.append(calc_relation(doc['bigram'], query))
    relations_idx = sorted(range(len(relations)), key=lambda k: relations[k])
    rankn = idx2file_name(relations_idx[-1:-101:-1])
    return rankn


if mp.cpu_count() >= 4:
    with mp.Pool(4) as pool:
        if args.feedback:
            results = pool.map(with_feedback, query_data)
        else:
            results = pool.map(test, query_data)
else:
    if args.feedback:
        results = pool.map(with_feedback, query_data)
    else:
        results = pool.map(test, query_data)
SubmitGenerator(results, args.ranked_list)
