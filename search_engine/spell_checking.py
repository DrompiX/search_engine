import re
from collections import Counter

from search_engine.utils import *


def build_dictionary(documents):
    """
    Build dictionary of original word forms (without stemming, but tokenized, lowercased, and only apt words considered)
    :param documents: dict of documents (contents)
    :return: {'word1': freq_word1, 'word2': freq_word2, ...}

    """
    result = Counter()

    for doc in documents:
        tokenized = tokenize(documents[doc].lower())
        for w in tokenized:
            if is_apt_word(w):
                result[w] += 1
    
    return dict(result)


def build_k_gram_index(dictionary, k):
    """
    Build index of k-grams for dictionary words. Padd with '$' ($word$) before splitting to k-grams
    :param dictionary: dictionary of original words
    :param k: number of symbols in one gram
    :return: {'gram1': ['word1_with_gram1', 'word2_with_gram1', ...],
              'gram2': ['word1_with_gram2', 'word2_with_gram2', ...], ...}
    """
    result = {}

    for word in dictionary.keys():
        w = '$' + word + '$'
        if len(w) >= k:
            for i in range(0, len(w) - k + 1):
                gram = w[i: i + k]
                i += k
                if gram not in result:
                    result[gram] = [word]
                else:
                    result[gram].append(word)
    
    return result
                

def generate_wildcard_options(wildcard, k_gram_index):
    """
    For a given wildcard return all words matching it using k-grams
    Refer to book chapter 3.2.2
    Don't forget to pad wildcard with '$', when appropriate
    :param wildcard: query word in a form of a wildcard
    :param k_gram_index:
    :return: list of options (matching words)
    """
    result = []

    k = len(list(k_gram_index.keys())[0])
    k_grams = build_k_gram_index({wildcard: 0}, k)

    wildcard = wildcard.replace('*', '.*')
    setlist = []
    
    for gram in k_grams:
        if gram in k_gram_index:
            matched = set()
            for word in k_gram_index[gram]:
                mathing = re.match(wildcard, word)
                if mathing and mathing.group(0) == word:
                    matched.add(word)
            
            if len(matched) > 0:
                setlist.append(matched)
    
    if len(setlist) > 0:
        result = list(set.intersection(*setlist))
    
    return result


def produce_soundex_code(word):
    """
    Implement soundex algorithm, version from book chapter 3.4
    :param word: word in lowercase
    :return: soundex 4-character code, like 'k450'
    """
    code = [word[0]]

    tranlation = '01230120022455012623010202'
    cur_digit = -1
    for char in word[1:]:
        digit = tranlation[ord(char) - ord('a')]
        if cur_digit == -1:
            cur_digit = digit

        if digit != cur_digit:
            if cur_digit != '0':
                code.append(str(cur_digit))
            cur_digit = digit
    
    if (code[-1] != cur_digit) and (cur_digit != '0') and cur_digit != -1:
        code.append(cur_digit)

    result = ''.join(code[:4]) + '0' * (4 - len(code))
    return result


def build_soundex_index(dictionary):
    """
    Build soundex index for dictionary words.
    :param dictionary: dictionary of original words
    :return: {'code1': ['word1_with_code1', 'word2_with_code1', ...],
              'code2': ['word1_with_code2', 'word2_with_code2', ...], ...}
    """
    result = {}

    for word in dictionary:
        code = produce_soundex_code(word)
        if code not in result:
            result[code] = [word]
        else:
            result[code].append(word)

    return result
