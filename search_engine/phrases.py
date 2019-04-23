import nltk
from nltk.collocations import *


def find_ngrams_PMI(tokenized_text, freq_thresh, pmi_thresh, n):
    """
    Finds n-grams in tokenized text, limiting by frequency and pmi value
    :param tokenized_text: list of tokens
    :param freq_thresh: number, only consider ngrams more frequent than this threshold
    :param pmi_thresh: number, only consider ngrams that have pmi value greater than this threshold
    :param n: length of ngrams to consider, can be 2 or 3
    :return: set of ngrams tuples - {('ngram1_1', 'ngram1_2'), ('ngram2_1', 'ngram2_2'), ... }
    """
    result = set()

    if n == 2:
        measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokenized_text)
    elif n == 3:
        measures = nltk.collocations.TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(tokenized_text)

    ngrams = finder.score_ngrams(measures.pmi)
    freqs = finder.ngram_fd
    for ngram in ngrams:
        print(ngram[0], ngram[1], freqs[ngram[0]])
        if (ngram[1] >= pmi_thresh) and (freqs[ngram[0]] >= freq_thresh):
            result.add(ngram[0])

    return result


def build_ngram_index(tokenized_documents, ngrams):
    """
    Builds index based on ngrams and collection of tokenized docs
    :param tokenized_documents: {doc1_id: ['token1', 'token2', ...], doc2_id: ['token1', 'token2', ...]}
    :param ngrams: set of ngrams tuples - {('ngram1_1', 'ngram1_2'), ('ngram2_1', 'ngram2_2', 'ngram2_3'), ... }
    :return: dictionary - {ngram_tuple :[ngram_tuple_frequency, (doc_id_1, doc_freq_1), (doc_id_2, doc_freq_2), ...], ...}
    """
    dictionary = {}

    doc_ngrams = {}
    for doc in tokenized_documents:
        ngrams_freq = {}

        measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokenized_documents[doc])
        freqs = finder.ngram_fd
        for ngram in freqs:
            ngrams_freq[ngram] = freqs[ngram]
        
        measures = nltk.collocations.TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(tokenized_documents[doc])
        freqs = finder.ngram_fd
        for ngram in freqs:
            ngrams_freq[ngram] = freqs[ngram]

        doc_ngrams[doc] = ngrams_freq

    for ngram in ngrams:
        dictionary[ngram] = [0]
        for doc in doc_ngrams:
            if ngram in doc_ngrams[doc]:
                dictionary[ngram][0] += doc_ngrams[doc][ngram]
                dictionary[ngram].append((doc, doc_ngrams[doc][ngram]))
    
    return dictionary
