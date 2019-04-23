import glob
import os
import pickle
import re
from collections import Counter

from bs4 import BeautifulSoup


def extract_categories(path):
    """
    Parses .sgm files in path folder wrt categories each document belongs to.
    Returns a list of documents for each category. One document usually belongs to several categories.
    Categories are contained in special tags (<TOPICS>, <PLACES>, etc.),
    see cat_descriptions_120396.txt file for details
    :param path: original data path
    :return: dict, category:[doc_id1, doc_id2, ...]
    """
    result = {}

    for filename in sorted(os.listdir(path)):
        if filename.endswith('.sgm'):
            with open(path + filename, 'r', encoding='latin1') as file:
                file_content = file.read()
                parsed = BeautifulSoup(file_content, 'html.parser')
                file_documents = parsed.find_all('reuters')
                
                for document in file_documents:
                    doc_id = int(document['newid'])
                    categories = [document.topics, document.places,
                                  document.people, document.orgs,
                                  document.exchanges, document.companies]
                    for cat in categories:
                        for doc_cat in cat.contents:
                            cat = doc_cat.string
                            if cat in result:
                                result[cat].append(doc_id)
                            else:
                                result[cat] = [doc_id]

    return result
    

def lm_rank_documents(query, doc_ids, doc_lengths, high_low_index, smoothing, param):
    """
    Scores each document in doc_ids using this document's language model.
    Applies smoothing. Looks up term frequencies in high_low_index
    :param query: dict, term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built last lab
    :param smoothing: which smoothing to apply, either 'additive' or 'jelinek-mercer'
    :param param: alpha for additive / lambda for jelinek-mercer
    :return: dictionary of scores, doc_id:score
    """
    result = {}

    if smoothing == 'additive':
        for doc_id in doc_ids:
            score = 1.0
            for term in query:
                cur_score = param
                denom = doc_lengths[doc_id] + param * len(high_low_index)
                if term in high_low_index:
                    if doc_id in high_low_index[term][0]:
                        cur_score += high_low_index[term][0][doc_id]
                    elif doc_id in high_low_index[term][1]:
                        cur_score += high_low_index[term][1][doc_id]
                
                score *= cur_score / denom

            result[doc_id] = score
    else:
        col_len = sum(doc_lengths.values())
        for doc_id in doc_ids:
            score = 1.0
            for term in query:
                cur_score = 0.0
                if term in high_low_index:
                    if doc_id in high_low_index[term][0]:
                        cur_score += high_low_index[term][0][doc_id]
                    elif doc_id in high_low_index[term][1]:
                        cur_score += high_low_index[term][1][doc_id]

                    cur_score = param * cur_score / doc_lengths[doc_id]
                    
                    high_freq = sum(high_low_index[term][0].values())
                    low_freq = sum(high_low_index[term][1].values())
    
                    cur_score += (1 - param) * (high_freq + low_freq) / col_len
                
                score *= cur_score
                
            result[doc_id] = score
    
    return result


def lm_define_categories(query, cat2docs, doc_lengths, high_low_index, smoothing, param):
    """
    Same as lm_rank_documents, but here instead of documents we score all categories
    to find out which of them the user is probably interested in. So, instead of building
    a language model for each document, we build a language model for each category -
    (category comprises all documents belonging to it)
    :param query: dict, term:count
    :param cat2docs: dict, category:[doc_id1, doc_id2, ...]
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built last lab
    :param smoothing: which smoothing to apply, either 'additive' or 'jelinek-mercer'
    :param param: alpha for additive / lambda for jelinek-mercer
    :return: dictionary of scores, category:score
    """
    result = {}

    if smoothing == 'additive':
        for cat in cat2docs:
            score = 1.0
            for term in query:
                tf, docs_len = 0, 0
                for doc_id in cat2docs[cat]:
                    docs_len += doc_lengths[doc_id]
                    if term in high_low_index:
                        if doc_id in high_low_index[term][0]:
                            tf += high_low_index[term][0][doc_id]
                        elif doc_id in high_low_index[term][1]:
                            tf += high_low_index[term][1][doc_id]
                
                score *= (param + tf) / docs_len + param * len(high_low_index)

            result[cat] = score
    else:
        col_len = sum(doc_lengths.values())
        for cat in cat2docs:
            score = 1.0
            for term in query:
                cur_score = 0.0
                tf, docs_len = 0, 0
                for doc_id in cat2docs[cat]:
                    docs_len += doc_lengths[doc_id]
                    if term in high_low_index:
                        if doc_id in high_low_index[term][0]:
                            tf += high_low_index[term][0][doc_id]
                        elif doc_id in high_low_index[term][1]:
                            tf += high_low_index[term][1][doc_id]

                if docs_len > 0:
                    cur_score = param * tf / docs_len

                high_freq = sum(high_low_index[term][0].values())
                low_freq = sum(high_low_index[term][1].values())
                cur_score += (1 - param) * (high_freq + low_freq) / col_len

                score *= cur_score
                
            result[cat] = score
    
    return result
