import pdb
import re
from utils import read_word_dict
import numpy as np
import xml.etree.ElementTree as ET
import unicodedata


def get_query(query_file, vocab_dict, partion=None):
    query_data = []
    file_path = query_file
    tree = ET.parse(file_path)
    root = tree.getroot()
    for child in root.findall("topic"):
        query = {0: 0}
        number = child.find("number").text
        query = str2bigram(query, child.find("title").text, vocab_dict)
        query = str2bigram(query, child.find("question").text[2:], vocab_dict)
        query = str2bigram(query, child.find("narrative").text[8:], vocab_dict)
        query = str2bigram(query, child.find("concepts").text, vocab_dict)
        query_data.append(query)
    return query_data


def str2unigram(query, data, vocab_dict):
    for word in split_word(data):
        if word in vocab_dict:
            if vocab_dict[word] in query:
                query[vocab_dict[word]] += 1
            else:
                query[vocab_dict[word]] = 1
        else:
            query[0] += 1
    return query


def str2bigram(query, data, vocab_dict):
    for word, next_word in zip(split_word(data)[:-1], split_word(data)[1:]):
        if word in vocab_dict and next_word in vocab_dict:
            if vocab_dict[word] in query:
                query[vocab_dict[word] + " " + vocab_dict[next_word]] += 1
            else:
                query[vocab_dict[word] + " " + vocab_dict[next_word]] = 1
        else:
            query[0] += 1
    return query


def split_word(string):
    string = unicodedata.normalize("NFKC", string)
    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(regex, string)
    return matches


# if __name__ == "__main__":
#     pdb.set_trace()
