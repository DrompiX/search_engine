import nltk
import re
from collections import Counter
from search_engine.utils import preprocess


def clean_text(text):
    '''Removes some trash from text and multiple spaces
    
    Args:
        text: input text
    Returns:
        clean text
    '''
    clean_text = re.sub(r'[’”“]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def get_text_sentences(text):
    '''Cleanes text and splits into sentences with nltk

    Args:
        text: text to be splitted into sentences
    
    Returns:
        list of sentences
    '''
    new_text = clean_text(text)
    sentences = nltk.sent_tokenize(new_text)
    return sentences


def naive_sum(doc, query, sentence_cnt):
    '''Implementaion of naive text summarization
    
    Idea is to take document data, preprocess it, divide into
    sentences, calculate score for each sentence based of tf
    of each word in it and multiply for tf of each word in 
    the query, and return top k sentences, for which
    sum of lengths is less or equal to `summary_len`

    Args:
        doc: text of the document
        query: input query
        sentence_cnt: max amount of terms for output
    
    Returns:
        resulting summary (title + summary text)
    '''
    sentences = get_text_sentences(doc)
    # calculating number of term occurences in query and text
    # TODO: check stemming difference
    q_tf = Counter(preprocess(query))
    tf = Counter(preprocess(doc))

    # normalizing tf on maximum tf
    max_freq = max(tf.values())
    for term in tf:
        tf[term] /= max_freq
    
    # calculating score for each sentence
    score_results = {}
    for sentence in sentences:
        for term in preprocess(sentence):
            if sentence in score_results:
                score_results[sentence] += tf[term] * q_tf[term]
            else:
                score_results[sentence] = tf[term] * q_tf[term]
    
    score_results = sorted(score_results.items(), key=lambda kv: kv[1], reverse=True)
    result = []
    for i in range(min(len(sentences), sentence_cnt)):
        result.append(score_results[i][0] + ' ')
    return ''.join(result)