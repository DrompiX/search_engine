import os

from search_engine import indexing
from search_engine import spell_checking

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
    }

    def __init__(self, scoring='okapi', paths=None):
        # scoring: 'okapi' or 'cosine'
        self.scoring =  _cosine_scoring if scoring == 'cosine' else _okapi_scoring
        self.index_built = self._is_built(self.index_paths)
    
    def index(self, path):
        if not self.index_built:
            indexing.build_inverted_index(path, self.index_paths)
        self.inv_index, self.doc_lengths, self.documents = indexing.load_index(self.index_paths)
        
        self.dictionary = spell_checking.build_dictionary(self.documents)
        self.k_gram_index = spell_checking.build_k_gram_index(self.dictionary, 2)
        self.soundex_index = spell_checking.build_soundex_index(self.dictionary)

        self.index_built = True
        pass

    def answer_query(self, raw_query, top_k, scoring_fnc):
        pass

    def _okapi_scoring(self):
        pass
    
    def _cosine_scoring(self):
        pass

    def _is_built(self, path: dict):
        if path is None:
            return False
        
        for f in path.values():
            if not os.path.isfile(f):
                return False
            
        return True
