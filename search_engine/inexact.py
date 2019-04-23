import math


def build_high_low_index(index, freq_thresh):
    """
    Build high-low index based on standard inverted index.
    Based on the frequency threshold, for each term doc_ids are are either put into "high list" -
    if term frequency in it is >= freq_thresh, or in "low list", otherwise.
    high_low_index should return a python dictionary, with terms as keys.
    The structure is different from that of standard index - for each term
    there is a list - [high_dict, low_dict, len(high_dict) + len(low_dict)],
    the latter is document frequency of a term. high_dict, as well as low_dict,
    are python dictionaries, with entries of the form doc_id : term_frequency
    :param index: inverted index
    :param freq_thresh: threshold on term frequency
    :return: dictionary
    """
    result = {}
    
    for term in index:
        result[term] = [{}, {}, -1]
        post_list = index[term]
        for i in range(1, len(post_list)):
            doc_id, freq = post_list[i]
            if freq >= freq_thresh:
                result[term][0][doc_id] = freq
            else:
                result[term][1][doc_id] = freq
        result[term][2] = len(result[term][0]) + len(result[term][1])
    
    return result
    

def filter_docs(query, high_low_index, min_n_docs):
    """
    Return a set of documents in which query terms are found.
    You are interested in getting the best documents for a query, therefore you
    will sequentially check for the following conditions and stop whenever you meet one.
    For each condition also check if number of documents is  >= min_n_docs.
    1) We consider only high lists for the query terms and return a set of documents such that each document contains
    ALL query terms.
    2) We search in both high and low lists, but still require that each returned document should contain ALL query terms.
    3) We consider only high lists for the query terms and return a set of documents such that each document contains
    AT LEAST ONE query term. Actually, a union of high sets.
    4) At this stage we are fine with both high and low lists, return a set of documents such that each of them contains
    AT LEAST ONE query term.

    :param query: dictionary term:count
    :param high_low_index: high-low index you built before
    :param min_n_docs: minimum number of documents we want to receive
    :return: set if doc_ids
    """
    result = set()
    
    started = False
    for term in query.keys():
        term_info = high_low_index[term]    
        if not started:
            result = set(term_info[0].keys())
            started = True
        else:
            result = result & set(term_info[0].keys())
    
    if len(result) >= min_n_docs:
        return result

    started = False
    for term in query.keys():
        term_info = high_low_index[term]    
        if not started:
            result = set(term_info[0].keys()) | set(term_info[1].keys()) 
            started = True
        else:
            result = result & (set(term_info[0].keys()) | set(term_info[1].keys()))
    
    if len(result) >= min_n_docs:
        return result

    started = False
    for term in query.keys():
        term_info = high_low_index[term]    
        if not started:
            result = set(term_info[0].keys())
            started = True
        else:
            result = result | set(term_info[0].keys())
    
    if len(result) >= min_n_docs:
        return result

    started = False
    for term in query.keys():
        term_info = high_low_index[term]    
        if not started:
            result = set(term_info[0].keys()) | set(term_info[1].keys()) 
            started = True
        else:
            result = result | (set(term_info[0].keys()) | set(term_info[1].keys()))
    
    if len(result) >= min_n_docs:
        return result



def cosine_scoring_docs(query, doc_ids, doc_lengths, high_low_index):
    """
    Change cosine_scoring function you built in the second lab
    such that you only score set of doc_ids you get as a parameter,
    and using high_low_index instead of standard inverted index
    :param query: dictionary term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built before
    :return: dictionary of scores, doc_id:score
    """
    scores = {}
    for term in query:
        idf = math.log10(len(doc_lengths) / (high_low_index[term][2]))
        for doc_id, freq in high_low_index[term][0].items():
            if doc_id in scores:
                scores[doc_id] += freq * query[term] * (idf ** 2)
            else:
                scores[doc_id] = freq * query[term] * (idf ** 2)

    for doc_id in scores:
        scores[doc_id] /= doc_lengths[doc_id]

    return scores


def okapi_scoring_docs(query, doc_ids, doc_lengths, high_low_index, k1=1.2, b=0.75):
    """
    Change okapi_scoring function you built in the second lab
    such that you only score set of doc_ids you get as a parameter,
    and using high_low_index instead of standard inverted index
    :param query: dictionary term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built before
    :return: dictionary of scores, doc_id:score
    """
    scores = {}
    avgdl = sum(doc_lengths.values()) / len(doc_lengths)
    for term in query:
        if term in high_low_index:
            idf = math.log10(len(doc_lengths) / (high_low_index[term][2]))
            for doc_id, freq in high_low_index[term][0].items():
                nominator = freq * (k1 + 1)
                denominator = (freq + k1 * (1 - b + b * doc_lengths[doc_id] / avgdl))
                if doc_id in scores:
                    scores[doc_id] += idf * nominator / denominator
                else:
                    scores[doc_id] = idf * nominator / denominator
    
    return scores
