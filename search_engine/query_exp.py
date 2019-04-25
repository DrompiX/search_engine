import numpy as np
from collections import Counter
from search_engine.utils import preprocess

def docs2vecs(docs, engine):
    '''Converts documents to vector representation
    
    Args:
        docs: documents to be converted
    Returns:
        resulting vectors
    '''
    vectors = {}
    for doc_id, doc in docs.items():
        terms = preprocess(str(doc))
        vectors[doc_id] = Counter()
        for term in terms:
            vectors[doc_id][term] += 1
            
    for doc_id in vectors:
        for term in vectors[doc_id]:
            if term in engine.inv_index:
                idf = np.log10(len(engine.documents) / engine.inv_index[term][0])
            else:
                idf = 0
            vectors[doc_id][term] *= idf
    
    return vectors


def rocchio(query, relevance, top_docs, engine, alph=1.0, beta=0.75, gamma=0.15):
    '''Implementation of Rocchio algorithm

    Args:
        query: input query
        relevance: list of relevant docs for query
        top_docs: top k docs for query
        alph: weight of original query
        beta: weight of relevant docs
        gamma: weight of irrelevant docs
    Return:
        modified query as Dict (term: score)
    '''
    top_docs_vectors = docs2vecs(top_docs, engine)
    query_vector = Counter(preprocess(query))
    new_query = dict((k, v * alph) for k, v in query_vector.items())
    
    # center for relevant docs
    center = dict((k, 0) for k in query_vector.keys()) 
    relevant_docs = set()
    for doc_id, _ in relevance:
        if doc_id in top_docs_vectors:
            relevant_docs.add(doc_id)
            for term in top_docs_vectors[doc_id]:
                if term in center:
                    center[term] += top_docs_vectors[doc_id][term]
                else:
                    center[term] = top_docs_vectors[doc_id][term]

    # center for irrelevant docs
    neg_center = dict((k, 0) for k in query_vector.keys())
    for doc_id in top_docs_vectors:
        if doc_id not in relevant_docs:
            for term in top_docs_vectors[doc_id]:
                if term in neg_center:
                    neg_center[term] += top_docs_vectors[doc_id][term]
                else:
                    neg_center[term] = top_docs_vectors[doc_id][term]
    
    # if no relevant docs, return same query
    if len(relevant_docs) == 0:
        return new_query

    term_candidates = {}
    # recalculate weights for terms and add new
    for term in center:
        if term in term_candidates:
        # if term in new_query:
            # new_query[term] += beta * 1 / len(relevant_docs) * center[term]
            term_candidates[term] += beta * 1 / len(relevant_docs) * center[term]
        else:
            # new_query[term] = beta * 1 / len(relevant_docs) * center[term]
            term_candidates[term] = beta * 1 / len(relevant_docs) * center[term]
    
    term_candidates = sorted(term_candidates.items(), key=lambda item: item[1], reverse=True)
    for term_score in term_candidates[:2]:
        new_query[term_score[0]] = term_score[1]

    non_rel_cnt = len(top_docs_vectors) - len(relevant_docs)
    if gamma > 0 and non_rel_cnt > 0:
        for term in neg_center:
            if term in new_query:
                new_query[term] -= gamma * 1 / non_rel_cnt * neg_center[term]
                new_query[term] = max(0, new_query[term])
    
    return new_query


def get_k_relevant_docs(docs, k):
    ''' Returns relevance for top k relevant docs

    Args:
        docs: considered docs
        k: amount of docs to return
    Returns:
        list of k tuples: (doc_id, 1)
    '''
    relevance = []
    relevant_cnt = min(int(len(docs) / 2), k)
    for doc_id in docs:
        if relevant_cnt == 0:
            break
        else:
            relevance.append((doc_id, 1))
            relevant_cnt -= 1
    
    return relevance


def pseudo_relevance_feedback(query, top_docs, engine, relevant_n=5, alph=1.0, beta=0.75, gamma=0):
    '''Implementation of pseudo relevance feedback
    
    Based on implementation of roccio algorithm

    Args:
        query: input query
        top_docs: top k docs for query
        relevant_n: number of first docs to consider relevant
        alph: weight of original query
        beta: weight of relevant docs
        gamma: weight of irrelevant docs
    Return:
        modified query as Dict (term: score)
    '''
    relevance = get_k_relevant_docs(top_docs, relevant_n)
    return rocchio(query, relevance, top_docs, engine, alph, beta, gamma)