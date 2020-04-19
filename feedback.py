from collections import Counter
from utils import load_pkl, get_rankn_bigram

ALPHA = 1
BETA = 0.75
GAMMA = -0.15


def rocchio_feedback(query, docs, pivot):
    p_centroid = find_centroid(docs[:pivot])
    n_centroid = find_centroid(docs[100-pivot:])
    query.update((x, y * ALPHA) for x, y in query.items())
    for key, value in p_centroid.items():
        if key in query:
            query[key] += value * BETA
        else:
            query[key] = value * BETA
    for key, value in n_centroid.items():
        if key in query:
            query[key] += value * GAMMA
        else:
            query[key] = value * GAMMA
    return query


def find_centroid(dicts):
    centroid = {}
    for d in dicts:
        for key, value in d.items():
            if key not in centroid:
                centroid[key] = value
            else:
                centroid[key] += value
    centroid.update((x, y/len(dicts)) for x, y in centroid.items())
    return centroid


if __name__ == "__main__":
    doc_datas = load_pkl("doc_datas.pkl")
    rankn_bigram = get_rankn_bigram([i for i in range(100)], doc_datas)
