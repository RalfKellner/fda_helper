from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
import sqlite3
from gensim.matutils import corpus2csc
import copy
import numpy as np
import pandas as pd
import pickle
import os

from typing import Callable
from abc import ABC, abstractmethod
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Corpus(ABC):
    def __init__(
            self, 
            preprocessor: Callable[[str], list] = simple_preprocess, 
            identify_phrases = True, 
            phrases_arguments = None,
            create_dictionary_and_countings = True, 
            dictionary_arguments = None, 
            tfidf_arguments_smart = None, 
            yield_documents = False,
            train_corpus = True
            ):
        self.preprocessor = preprocessor
        self.n_documents = self.__len__()
        self.has_phrases_model = False
        self.identify_phrases = identify_phrases
        if phrases_arguments == None:
            self.phrases_arguments = {
                "min_count": 25,
                "threshold": 20
            }
        self.create_dictionary_and_countings = create_dictionary_and_countings
        self.dictionary_arguments = dictionary_arguments
        if self.dictionary_arguments == None:
            self.dictionary_arguments = {
                "no_below": int(np.ceil(0.01 * self.n_documents)),
                "no_above": 0.75,
                "keep_n": 2000
            }
        self.tfidf_arguments_smart = tfidf_arguments_smart
        if self.tfidf_arguments_smart == None:
            self.tfidf_arguments_smart = "ntn"
        self.yield_documents = yield_documents
        self.logger = logging.getLogger(self.__class__.__name__)
        self.train_corpus = train_corpus

        self.phrases_model = None
        self.dictionary = None
        self.tfidf_model = None

        if self.train_corpus:
            if self.identify_phrases:
                self.logger.info("Starting to identify phrases.")
                self._learn_phrases()
                self.logger.info("Phrases have been identified. Building dictionary.")
            if self.create_dictionary_and_countings:
                self.dictionary = Dictionary(self._yield_tokens())
                self.dictionary.filter_extremes(**self.dictionary_arguments)
                self.logger.info(f"Filtered words from bag-of-words dictionary if the words occur in less than {int(np.ceil(0.01 * self.n_documents))} or in more than 75% of all documents.")
                self.logger.info("Dictionary has been build. Creating bag-of-words and tfidf-representations.")
                self.tfidf_model = TfidfModel(self.bow(), smartirs = self.tfidf_arguments_smart)
                self.logger.info("Bag-of-words and tfidf vectors are ready.")

    @abstractmethod
    def _document_loader():
        pass

    def _yield_tokens(self):
        for doc in self._document_loader():
            if self.has_phrases_model:
                yield self.phrases_model[self.preprocessor(doc)]
            else:
                yield self.preprocessor(doc)

    def _yield_documents(self):
        counter = -1
        for doc in self._document_loader():
            counter += 1
            if self.has_phrases_model:
                yield TaggedDocument(self.phrases_model[self.preprocessor(doc)], [counter])
            else:
                yield TaggedDocument(self.preprocessor(doc), [counter])

    def _learn_phrases(self):
        self.logger.info(f"Phrases are identified using a min_count of {25} and a scoring threshold of {20}.")
        self.phrases_model = Phrases(self._yield_tokens(), min_count = 25, threshold = 20, connector_words=ENGLISH_CONNECTOR_WORDS)
        self.has_phrases_model = True

    def bow(self):
        for doc in self._yield_tokens():
            yield self.dictionary.doc2bow(doc)

    def bow_to_matrix(self, sparse = True):
        bow_matrix = corpus2csc(self.bow(), num_terms = len(self.dictionary)).transpose()
        if sparse:
            return bow_matrix
        else:
            return bow_matrix.toarray()

    def tfidf_to_matrix(self, sparse = True):
        tfidf = self.tfidf_model[self.bow()]
        tfidf_matrix = corpus2csc(tfidf, num_terms = len(self.dictionary)).transpose()
        if sparse:
            return tfidf_matrix
        else:
            return tfidf_matrix.toarray()
        
    def get_dictionary_vobulary(self):
        if self.dictionary:
            _ = self.dictionary.most_common()
            return list(self.dictionary.id2token.values())
  
    def save(self, file_path):
        data = self.__dict__.copy()
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"Model saved to {file_path}")

    @classmethod
    @abstractmethod
    def load(cls):
        pass

    def copy(self):
        return copy.deepcopy(self)
    
    def __iter__(self):
        if self.yield_documents:
            for doc in self._yield_documents():
                yield doc
        else:
            for doc in self._yield_tokens():
                yield doc

    @abstractmethod
    def __len__(self):
        return sum(1 for _ in self._document_loader())
    

class SQLiteCorpus(Corpus):
    def __init__(
            self, 
            db_name,
            sql_query,
            preprocessor: Callable[[str], list] = simple_preprocess, 
            identify_phrases = True, 
            phrases_arguments = None,
            create_dictionary_and_countings = True, 
            dictionary_arguments = None, 
            tfidf_arguments_smart = None, 
            yield_documents = False,
            train_corpus = True
        ):
        self.db_name = db_name
        self.sql_query = sql_query
        super().__init__(
            preprocessor = preprocessor, 
            identify_phrases = identify_phrases, 
            phrases_arguments = phrases_arguments,
            create_dictionary_and_countings = create_dictionary_and_countings, 
            dictionary_arguments = dictionary_arguments,
            tfidf_arguments_smart = tfidf_arguments_smart,
            yield_documents = yield_documents,
            train_corpus = train_corpus)
        
    def _document_loader(self):
        conn = sqlite3.connect(self.db_name)
        res = conn.execute(self.sql_query)
        for row in res:
            yield row[0]
        conn.close()

    def update_sql_query(self, sql_query):
        self.sql_query = sql_query
        self.n_documents = self.__len__()

    def collect_dataframe_from_db(self, sql_query):
        conn = sqlite3.connect(self.db_name)
        df = pd.read_sql(sql_query, conn)
        conn.close()
        return df

    @classmethod
    def load(cls, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            db_name = data["db_name"],
            sql_query = data["sql_query"],
            preprocessor = data["preprocessor"], 
            identify_phrases = data["identify_phrases"], 
            phrases_arguments = data["phrases_arguments"],
            create_dictionary_and_countings = data["create_dictionary_and_countings"],
            dictionary_arguments = data["dictionary_arguments"],
            tfidf_arguments_smart = data["tfidf_arguments_smart"],
            yield_documents = data["yield_documents"],
            train_corpus = False
        )

        for key, value in data.items():
            if key not in {
                "db_name", 
                "sql_query", 
                "preprocessor", 
                "identify_phrases", 
                "phrases_arguments", 
                "create_dictionary_and_countings", 
                "dictionary_arguments", 
                "tfidf_arguments_smart", 
                "yield_documents", 
                "train_corpus"
                }: setattr(instance, key, value)

        return instance   

    def __len__(self):
        return sum(1 for _ in self._document_loader())
    

class ListCorpus(Corpus):
    def __init__(
            self, 
            documents,
            preprocessor: Callable[[str], list] = simple_preprocess, 
            identify_phrases = True, 
            phrases_arguments = None,
            create_dictionary_and_countings = True, 
            dictionary_arguments = None, 
            tfidf_arguments_smart = None, 
            yield_documents = False,
            train_corpus = True
        ):
        self.documents = documents
        super().__init__(
            preprocessor = preprocessor, 
            identify_phrases = identify_phrases, 
            phrases_arguments = phrases_arguments,
            create_dictionary_and_countings = create_dictionary_and_countings,
            dictionary_arguments = dictionary_arguments,
            tfidf_arguments_smart = tfidf_arguments_smart, 
            yield_documents = yield_documents,
            train_corpus = train_corpus
            )
        
    def _document_loader(self):
        for doc in self.documents:
            yield doc

    @classmethod
    def load(cls, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            documents = data["documents"],
            preprocessor = data["preprocessor"], 
            identify_phrases = data["identify_phrases"], 
            phrases_arguments = data["phrases_arguments"],
            create_dictionary_and_countings = data["create_dictionary_and_countings"],
            dictionary_arguments = data["dictionary_arguments"],
            tfidf_arguments_smart = data["tfidf_arguments_smart"],
            yield_documents = data["yield_documents"],
            train_corpus = False
        )

        for key, value in data.items():
            if key not in {
                "documents", 
                "preprocessor", 
                "identify_phrases", 
                "phrases_arguments", 
                "create_dictionary_and_countings", 
                "dictionary_arguments", 
                "tfidf_arguments_smart", 
                "yield_documents", 
                "train_corpus"
                }: setattr(instance, key, value)

        return instance    

    def __len__(self):
        return len(self.documents)
    
