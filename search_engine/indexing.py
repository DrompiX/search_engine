import os
import pickle
from bs4 import BeautifulSoup
from search_engine.utils import preprocess

def build_inverted_index(path, save_paths):
    """
    # principal function - builds an index of terms in all documents
    # generates 3 dictionaries and saves on disk as separate files:
    # index - term:[term_frequency, (doc_id_1, doc_freq_1), (doc_id_2, doc_freq_2), ...]
    # doc_lengths - doc_id:doc_length
    # documents - doc_id: doc_content_clean
    :param path: path to directory with original reuters files
    :param limit: number of articles to process, for testing. If limit is not None,
                  return index when done, without writing files to disk
    """
    print('Building index...')
    index = {}
    doc_lengths = {}
    documents = {}
    
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.sgm'):
            with open(path + filename, 'r', encoding='latin1') as file:
                file_content = file.read()
                parsed = BeautifulSoup(file_content, 'html.parser')
                file_documents = parsed.find_all('reuters')

                for document in file_documents:
                    doc_id = int(document['newid'])
                    doc_title = document.title.text if document.title else ''
                    doc_body = document.body.text if document.body else ''

                    ext_document = ''
                    if (doc_title == '') or (doc_body == ''):
                        ext_document = doc_title + doc_body
                    else:
                        ext_document = doc_title + '\n' + doc_body
                    documents[doc_id] = ext_document

                    doc_terms = preprocess(ext_document)
                    doc_lengths[doc_id] = len(doc_terms)

                    tf = {}
                    for term in doc_terms:
                        if term in tf:
                            tf[term] += 1
                        else:
                            tf[term] = 1
                    
                    for term in tf:
                        if term in index:
                            index[term][0] += 1
                        else:
                            index[term] = [1]
                        index[term].append((doc_id, tf[term]))
                    
    with open(save_paths['inv_index'], 'wb') as dump_file:
        pickle.dump(index, dump_file)
    
    with open(save_paths['doc_lengths'], 'wb') as dump_file:
        pickle.dump(doc_lengths, dump_file)

    with open(save_paths['documents'], 'wb') as dump_file:
        pickle.dump(documents, dump_file)
    
    print('Index was built!')
    

def load_index(save_paths):
    print('Loading index...')
    with open(save_paths['inv_index'], 'rb') as fp:
        index = pickle.load(fp)
    
    with open(save_paths['doc_lengths'], 'rb') as fp:
        doc_lengths = pickle.load(fp)

    with open(save_paths['documents'], 'rb') as fp:
        documents = pickle.load(fp)
    print('Index was loaded!')
    return index, doc_lengths, documents