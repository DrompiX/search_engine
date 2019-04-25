import os
import pickle
import math
import heapq
import re
import time
from functools import partial
from collections import Counter

from search_engine import indexing
from search_engine import spell_checking
from search_engine import inexact
from search_engine import language_model
from search_engine import query_exp
from search_engine import phrases
from search_engine.doc_sum import naive_sum
from search_engine.utils import *

path_prefix = 'search_engine/data.nosync/'


class SearchEngine(object):

    index_paths = {
        'inv_index': f'{path_prefix}inv_index.p',
        'documents': f'{path_prefix}documents.p',
        'doc_lengths': f'{path_prefix}doc_lengths.p'
    }

    sc_paths = {
        'k_gram_index': f'{path_prefix}k_gram_index.p',
        'dictionary': f'{path_prefix}dictionary.p',
        'soundex': f'{path_prefix}soundex.p'
    }

    inexact_paths = {
        'high_low_index': f'{path_prefix}high_low_index.p'
    }

    phrase_paths = {
        'n_gram_index': f'{path_prefix}n_gram_index.p'
    }

    def __init__(self, paths=None):
        self.index_built = self._is_built(self.index_paths)
    
    def do_indexing(self, path):
        if not self.index_built:
            indexing.build_inverted_index(path, self.index_paths)
        self.inv_index, self.doc_lengths, self.documents = indexing.load_index(self.index_paths)
        
        self.dictionary = self._load_dictionary(self.sc_paths['dictionary'])
        self.k_gram_index = self._load_k_gram_index(self.sc_paths['k_gram_index'])
        self.soundex_index = self._load_soundex(self.sc_paths['soundex'])

        self.high_low_index = self._load_high_low_index(self.inexact_paths['high_low_index'])

        self.n_gram_index = self._load_n_gram_index(self.phrase_paths['n_gram_index'])
        self.index_built = True

    def _handle_wildcards(self, raw_query):
        for word in tokenize(raw_query.lower()):
            if word.find('*') != -1:
                return spell_checking.generate_wildcard_options(word, self.k_gram_index)
        return []
    
    def _handle_soundex(self, query):
        errors = {}
        for word in query:
            if word not in self.inv_index:
                word_code = spell_checking.produce_soundex_code(word)
                if word_code in self.soundex_index:
                    for corr in self.soundex_index[word_code]:
                        if word in errors:
                            errors[word].append(corr)
                        else:
                            errors[word] = [corr]
        
        return errors

    def _answer_inexact(self, query, top_k, scoring='okapi'):
        doc_ids = inexact.filter_docs(query, self.high_low_index, top_k)
        if scoring == 'lm':
            score_fun = partial(language_model.lm_rank_documents, 
                                smoothing='additive', param=0.1)
        elif scoring == 'cosine':
            score_fun = inexact.cosine_scoring_docs
        else:
            score_fun = inexact.okapi_scoring_docs
        return score_fun(query, doc_ids, self.doc_lengths, self.high_low_index)

    def _select_scoring_fun(self, scoring):
        if scoring == 'lm':
            return language_model.lm_rank_documents
        elif scoring == 'cosine':
            return inexact.cosine_scoring_docs
        else:
            return inexact.okapi_scoring_docs

    def answer_query(self, raw_query, top_k, do_inexact=False, scoring='okapi', 
                     summary_len=5, use_expansion=False, is_raw=True, do_phrase=False):
        start_time = time.time()
        score_fun =  self._cosine_scoring if scoring == 'cosine' else self._okapi_scoring
        # scoring = self._select_scoring_fun(scoring)
        # count frequency
        if is_raw:
            query = preprocess(raw_query)
            # ngrams_query = phrases.find_ngrams_PMI(query, 0, 1, 2)
            # ngrams_query |= phrases.find_ngrams_PMI(query, 0, 1, 3)
            query = Counter(query)
            
            wcs = self._handle_wildcards(raw_query)
            if len(wcs) != 0:
                print('\033[92mDid you mean:\033[0m')
                print(*wcs, sep=', ', end='?')
                return []
            
            sx = self._handle_soundex(query)
            if len(sx) != 0:
                print('\033[92mPossible soundex fixes:\033[0m')
                for w, corr in sx.items():
                    print(f'{w} -> ', end='')
                    print(*corr, sep=', ')
            
        else:
            query = raw_query

        if do_inexact:
            scores = self._answer_inexact(query, int(top_k / 5), scoring)
        elif do_phrase and is_raw:
            _query = preprocess(raw_query)
            ngrams_query = phrases.find_ngrams_PMI(_query, 0, 1, 2)
            ngrams_query |= phrases.find_ngrams_PMI(_query, 0, 1, 3)
            ngrams_query = dict((k, 1) for k in ngrams_query)
            scores = self._cosine_scoring_phrase(ngrams_query)
        else:
            scores = score_fun(query)

        
        h = []
        for doc_id in scores.keys():
            neg_score = -scores[doc_id]
            heapq.heappush(h, (neg_score, doc_id))
        
        # retrieve best matches
        top_k = min(top_k, len(h))  # handling the case when less than top k results are returned
        if is_raw:
            print('\033[1m\033[94mANSWERING TO:', raw_query, 'METHOD:', scoring, '\033[0m')
        else:
            print('\033[1m\033[94mANSWERING TO:', ' '.join(raw_query.keys()), 'METHOD:', scoring, '\033[0m')
        print(top_k, "results retrieved")
        top_k_ids = []
        articles = []
        for k in range(top_k):
            best_so_far = heapq.heappop(h)
            top_k_ids.append(best_so_far)
            article = naive_sum(self.documents[best_so_far[1]], raw_query, summary_len, is_raw)
            article_terms = tokenize(article)
            intersection = [t for t in article_terms if is_apt_word(t) and stem(t, ps) in query.keys()]
            for term in intersection:  # highlight terms for visual evaluation
                article = re.sub(r'(' + term + ')', r'\033[1m\033[91m\1\033[0m', article, flags=re.I)
            articles.append(article)
        print(top_k_ids)
        if use_expansion:
            id2doc = dict((k, v) for k, v in zip(top_k_ids, articles))
            new_query = query_exp.pseudo_relevance_feedback(raw_query, id2doc, self, relevant_n=2)
            return self.answer_query(new_query, top_k, do_inexact, scoring, summary_len, 
                                     use_expansion=False, is_raw=False)

        for article in articles:
            print("-------------------------------------------------------")
            print(article)

        print("\n--- Query executed in %.7s seconds ---\n" % (time.time() - start_time))
        
        return top_k_ids

    def _okapi_scoring(self, query, k1=1.2, b=0.75):
        """
        Computes scores for all documents containing any of query terms
        according to the Okapi BM25 ranking function, refer to wikipedia,
        but calculate IDF as described in chapter 6, using 10 as a base of log

        :param query: dictionary - term:frequency
        :return: dictionary of scores - doc_id:score
        """
        scores = Counter()
        avgdl = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        for term in query:
            if term in self.inv_index:
                idf = math.log10(len(self.doc_lengths) / (len(self.inv_index[term]) - 1))
                for i in range(1, len(self.inv_index[term])):
                    doc_id, doc_freq = self.inv_index[term][i]
                    nominator = doc_freq * (k1 + 1)
                    denominator = (doc_freq + k1 * (1 - b + b * self.doc_lengths[doc_id] / avgdl))
                    scores[doc_id] += idf * nominator / denominator
        
        return dict(scores)
    
    def _cosine_scoring(self, query):
        """
        Computes scores for all documents containing any of query terms
        according to the COSINESCORE(q) algorithm from the book (chapter 6)

        :param query: dictionary - term:frequency
        :return: dictionary of scores - doc_id:score
        """
        scores = Counter()
        for term in query:
            idf = math.log10(len(self.doc_lengths) / (len(self.inv_index[term]) - 1))
            for i in range(1, len(self.inv_index[term])):
                doc_id, doc_freq = self.inv_index[term][i]
                scores[doc_id] += doc_freq * query[term] * idf * idf

        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]

        return dict(scores)
    
    def _cosine_scoring_phrase(self, query):
        """
        Computes scores for all documents containing phrase from query terms

        :param query: string - raw query
        :return: dictionary of scores - doc_id:score
        """
        scores = Counter()
        for term in query:
            idf = math.log10(len(self.doc_lengths) / (len(self.n_gram_index[term]) - 1))
            for i in range(1, len(self.n_gram_index[term])):
                doc_id, doc_freq = self.n_gram_index[term][i]
                scores[doc_id] += doc_freq * query[term] * idf * idf

        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]

        return dict(scores)

    def _is_built(self, path: dict):
        if path is None:
            return False
        
        for f in path.values():
            if not os.path.isfile(f):
                return False
            
        return True
    
    def _load_dictionary(self, path):
        dictionary = self._load(path)
        if not dictionary:
            dictionary = spell_checking.build_dictionary(self.documents)
            self._save(dictionary, path)
        return dictionary

    def _load_k_gram_index(self, path):
        index = self._load(path)
        if not index:
            index = spell_checking.build_k_gram_index(self.dictionary, 2)
            self._save(index, path)
        return index

    def _load_soundex(self, path):
        soundex = self._load(path)
        if not soundex:
            soundex = spell_checking.build_soundex_index(self.dictionary)
            self._save(soundex, path)
        return soundex
    
    def _load_high_low_index(self, path):
        high_low = self._load(path)
        if not high_low:
            high_low = inexact.build_high_low_index(self.inv_index, 5)
            self._save(high_low, path)
        return high_low
    
    def _load_n_gram_index(self, path):
        index = self._load(path)
        if not index:
            ngrams = set()
            docs = {}
            for doc_id, doc in self.documents.items():
                prep_doc = preprocess(doc)
                docs[doc_id] = prep_doc
                n2grams = phrases.find_ngrams_PMI(prep_doc, 2, 6, 2)
                n3grams = phrases.find_ngrams_PMI(prep_doc, 2, 12, 3)
                ngrams = ngrams | (n2grams | n3grams)
            
            index = phrases.build_ngram_index(docs, ngrams)
            self._save(index, path)
        return index
    
    def _save(self, data, path):
        print(f'Saving {path}')
        with open(path, 'wb') as fd:
            pickle.dump(data, fd)
    
    def _load(self, path):
        result = None
        if os.path.isfile(path):
            print(f'Loading {path}')
            with open(path, 'rb') as fd:
                result = pickle.load(fd)
        return result
