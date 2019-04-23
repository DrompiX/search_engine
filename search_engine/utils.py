import nltk
# nltk.download('punkt')

stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
ps = nltk.stem.PorterStemmer()


def tokenize(text):
    return nltk.word_tokenize(text)


def stem(word, stemmer):
    return stemmer.stem(word)


def is_apt_word(word):
    return word not in stop_words and word.isalpha()


def preprocess(text, use_stem=True):
    tokenized = tokenize(text.lower())
    if use_stem:
        return [stem(w, ps) for w in tokenized if is_apt_word(w)]
    else:
        return [w for w in tokenized if is_apt_word(w)]