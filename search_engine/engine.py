import os
import pickle
import math
import heapq
import re
from collections import Counter

from search_engine import indexing
from search_engine import spell_checking
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

    def __init__(self, paths=None):
        # self.scoring =  self._cosine_scoring if scoring == 'cosine' else self._okapi_scoring
        self.index_built = self._is_built(self.index_paths)
    
    def do_indexing(self, path):
        if not self.index_built:
            indexing.build_inverted_index(path, self.index_paths)
        self.inv_index, self.doc_lengths, self.documents = indexing.load_index(self.index_paths)
        
        self.dictionary = self._load_dictionary(self.sc_paths['dictionary'])
        self.k_gram_index = self._load_k_gram_index(self.sc_paths['k_gram_index'])
        self.soundex_index = self._load_soundex(self.sc_paths['soundex'])

        self.index_built = True
        pass

    def answer_query(self, raw_query, top_k, scoring='okapi'):
        scoring =  self._cosine_scoring if scoring == 'cosine' else self._okapi_scoring
        query = preprocess(raw_query)
        # count frequency
        query = Counter(query)
        # retrieve all scores
        scores = scoring(query)
        # put them in heapq data structure, to allow convenient extraction of top k elements
        h = []
        for doc_id in scores.keys():
            neg_score = -scores[doc_id]
            heapq.heappush(h, (neg_score, doc_id))
        # retrieve best matches
        top_k = min(top_k, len(h))  # handling the case when less than top k results are returned
        print('\033[1m\033[94mANSWERING TO:', raw_query, 'METHOD:', scoring.__name__, '\033[0m')
        print(top_k, "results retrieved")
        top_k_ids = []
        for k in range(top_k):
            best_so_far = heapq.heappop(h)
            top_k_ids.append(best_so_far)
            article = self.documents[best_so_far[1]]
            article_terms = tokenize(article)
            intersection = [t for t in article_terms if is_apt_word(t) and stem(t, ps) in query.keys()]
            for term in intersection:  # highlight terms for visual evaluation
                article = re.sub(r'(' + term + ')', r'\033[1m\033[91m\1\033[0m', article, flags=re.I)
            print("-------------------------------------------------------")
            print(article)

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
