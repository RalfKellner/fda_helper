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
from typing import Callable, Dict, Generator, List, Optional, Union, Any
from abc import ABC, abstractmethod
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Corpus(ABC):
    """
    Abstract base class for a text corpus pipeline that preprocesses, 
    builds dictionaries, applies TF-IDF, and handles phrase identification.
    """

    def __init__(
        self,
        preprocessor: Callable[[str], List[str]] = simple_preprocess,
        identify_phrases: bool = True,
        phrases_arguments: Optional[Dict[str, Any]] = None,
        create_dictionary_and_countings: bool = True,
        dictionary_arguments: Optional[Dict[str, Union[int, float]]] = None,
        tfidf_arguments_smart: Optional[str] = None,
        yield_documents: bool = False,
        train_corpus: bool = True
    ) -> None:
        """
        Initialize the Corpus object with preprocessing and optional models.

        Args:
            preprocessor (Callable[[str], List[str]]): Tokenization and preprocessing function.
            identify_phrases (bool): Whether to detect phrases in the corpus.
            phrases_arguments (Optional[Dict[str, Any]]): Parameters for the Phrases model. See https://radimrehurek.com/gensim/models/phrases.html
            create_dictionary_and_countings (bool): Whether to create a dictionary and apply filters.
            dictionary_arguments (Optional[Dict[str, Union[int, float]]]): Parameters for dictionary filtering. See https://radimrehurek.com/gensim/corpora/dictionary.html and the method filter_extremes
            tfidf_arguments_smart (Optional[str]): Scoring scheme for TF-IDF ("ntn", "nnn", etc.). See https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
            yield_documents (bool): Whether to yield TaggedDocument objects. Only needed when we want to train a Doc2Vec model.
            train_corpus (bool): Whether to initialize models and preprocess the corpus. Set to false when loading and if we want to use dictionaries, phrases, tfidf, etc. from training corpus for test corpus
        """
        self.preprocessor = preprocessor
        self.n_documents = self.__len__()
        self.has_phrases_model = False
        self.identify_phrases = identify_phrases
        self.phrases_arguments = phrases_arguments or {"min_count": 25, "threshold": 20}
        self.create_dictionary_and_countings = create_dictionary_and_countings
        self.dictionary_arguments = dictionary_arguments or {
            "no_below": int(np.ceil(0.01 * self.n_documents)),
            "no_above": 0.75,
            "keep_n": 2000
        }
        self.tfidf_arguments_smart = tfidf_arguments_smart or "ntn"
        self.yield_documents = yield_documents
        self.train_corpus = train_corpus
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize models
        self.phrases_model: Optional[Phrases] = None
        self.dictionary: Optional[Dictionary] = None
        self.tfidf_model: Optional[TfidfModel] = None

        if self.train_corpus:
            if self.identify_phrases:
                self.logger.info("Starting to identify phrases.")
                self._learn_phrases()
                self.logger.info("Phrases have been identified. Building dictionary.")
            if self.create_dictionary_and_countings:
                self.dictionary = Dictionary(self._yield_tokens())
                self.dictionary.filter_extremes(**self.dictionary_arguments)
                self.logger.info(
                    f"Filtered words from bag-of-words dictionary if they occur in less than "
                    f"{self.dictionary_arguments['no_below']} documents or in more than "
                    f"{self.dictionary_arguments['no_above'] * 100}% of all documents."
                )
                self.logger.info("Dictionary has been built. Creating bag-of-words and TF-IDF representations.")
                self.tfidf_model = TfidfModel(self.bow(), smartirs=self.tfidf_arguments_smart)
                self.logger.info("Bag-of-words and TF-IDF vectors are ready.")

    @abstractmethod
    def _document_loader(self) -> Generator[str, None, None]:
        """
        Abstract method to load documents. Must be implemented in subclasses.
        
        Yields:
            str: A document as a string.
        """
        pass

    def _yield_tokens(self) -> Generator[List[str], None, None]:
        """Yield preprocessed tokens from documents."""
        for doc in self._document_loader():
            if self.has_phrases_model:
                yield self.phrases_model[self.preprocessor(doc)]
            else:
                yield self.preprocessor(doc)

    def _yield_documents(self) -> Generator[TaggedDocument, None, None]:
        """Yield TaggedDocument objects for training models."""
        counter = -1
        for doc in self._document_loader():
            counter += 1
            if self.has_phrases_model:
                yield TaggedDocument(self.phrases_model[self.preprocessor(doc)], [counter])
            else:
                yield TaggedDocument(self.preprocessor(doc), [counter])

    def _learn_phrases(self) -> None:
        """Train the Phrases model on the corpus."""
        self.logger.info(
            f"Phrases are identified using a min_count of {self.phrases_arguments['min_count']} "
            f"and a scoring threshold of {self.phrases_arguments['threshold']}."
        )
        self.phrases_model = Phrases(self._yield_tokens(), **self.phrases_arguments)
        self.has_phrases_model = True

    def bow(self) -> Generator[List[tuple], None, None]:
        """Yield bag-of-words representations of the documents."""
        for doc in self._yield_tokens():
            yield self.dictionary.doc2bow(doc)

    def bow_to_matrix(self, sparse: bool = True) -> Union[np.ndarray, Any]:
        """
        Convert the bag-of-words corpus into a matrix.

        Args:
            sparse (bool): Whether to return a sparse matrix.

        Returns:
            Union[np.ndarray, Any]: Document-term matrix in sparse or dense format.
        """
        bow_matrix = corpus2csc(self.bow(), num_terms=len(self.dictionary)).transpose()
        return bow_matrix if sparse else bow_matrix.toarray()

    def tfidf_to_matrix(self, sparse: bool = True) -> Union[np.ndarray, Any]:
        """
        Convert the TF-IDF corpus into a matrix.

        Args:
            sparse (bool): Whether to return a sparse matrix.

        Returns:
            Union[np.ndarray, Any]: TF-IDF matrix in sparse or dense format.
        """
        tfidf = self.tfidf_model[self.bow()]
        tfidf_matrix = corpus2csc(tfidf, num_terms=len(self.dictionary)).transpose()
        return tfidf_matrix if sparse else tfidf_matrix.toarray()

    def get_dictionary_vocabulary(self) -> Optional[List[str]]:
        """
        Retrieve the vocabulary from the dictionary.

        Returns:
            Optional[List[str]]: List of vocabulary terms.
        """
        if self.dictionary:
            return list(self.dictionary.token2id.keys())
        return None

    def save(self, file_path: str) -> None:
        """
        Save the current state of the Corpus object.

        Args:
            file_path (str): Path to save the serialized object.
        """
        data = self.__dict__.copy()
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"Model saved to {file_path}")

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> 'Corpus':
        """
        Load a saved Corpus object. Must be implemented for each subclass as they differ in input arguments. 

        Args:
            file_path (str): Path to the saved object.

        Returns:
            Corpus: The deserialized Corpus object.
        """
        pass

    def copy(self) -> 'Corpus':
        """
        Create a deep copy of the current object. Especially useful if we want to use models from training data for a test corpus.

        Returns:
            Corpus: A new deep copy of the Corpus object.
        """
        return copy.deepcopy(self)

    def __iter__(self) -> Generator[Union[List[str], TaggedDocument], None, None]:
        """
        Iterate over the documents in the corpus.

        Yields:
            Union[List[str], TaggedDocument]: Preprocessed tokens or TaggedDocument objects.
        """
        if self.yield_documents:
            yield from self._yield_documents()
        else:
            yield from self._yield_tokens()

    @abstractmethod
    def __len__(self) -> int:
        """
        Abstract method to get the number of documents. Must be implemented in subclasses.

        Returns:
            int: Number of documents in the corpus.
        """
        return sum(1 for _ in self._document_loader())
    

class SQLiteCorpus(Corpus):
    """
    SQLite-backed implementation of the Corpus class, enabling document processing directly from a database.
    """

    def __init__(
        self,
        db_name: str,
        sql_query: str,
        preprocessor: Callable[[str], List[str]] = simple_preprocess,
        identify_phrases: bool = True,
        phrases_arguments: Optional[Dict[str, Any]] = None,
        create_dictionary_and_countings: bool = True,
        dictionary_arguments: Optional[Dict[str, Union[int, float]]] = None,
        tfidf_arguments_smart: Optional[str] = None,
        yield_documents: bool = False,
        train_corpus: bool = True
    ) -> None:
        """
        Initialize an SQLiteCorpus object for processing documents from a database.

        Args:
            db_name (str): Path to the SQLite database file.
            sql_query (str): SQL query to fetch documents from the database. Must be defined such that the first column selected is the one with text documents.
            preprocessor (Callable[[str], List[str]]): Tokenization and preprocessing function.
            identify_phrases (bool): Whether to detect phrases in the corpus.
            phrases_arguments (Optional[Dict[str, Any]]): Parameters for the Phrases model. See https://radimrehurek.com/gensim/models/phrases.html
            create_dictionary_and_countings (bool): Whether to create a dictionary and apply filters.
            dictionary_arguments (Optional[Dict[str, Union[int, float]]]): Parameters for dictionary filtering. See https://radimrehurek.com/gensim/corpora/dictionary.html and the method filter_extremes
            tfidf_arguments_smart (Optional[str]): Scoring scheme for TF-IDF ("ntn", "nnn", etc.). See https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
            yield_documents (bool): Whether to yield TaggedDocument objects. Only needed when we want to train a Doc2Vec model.
            train_corpus (bool): Whether to initialize models and preprocess the corpus.
        """
        self.db_name = db_name
        self.sql_query = sql_query
        super().__init__(
            preprocessor=preprocessor,
            identify_phrases=identify_phrases,
            phrases_arguments=phrases_arguments,
            create_dictionary_and_countings=create_dictionary_and_countings,
            dictionary_arguments=dictionary_arguments,
            tfidf_arguments_smart=tfidf_arguments_smart,
            yield_documents=yield_documents,
            train_corpus=train_corpus
        )

    def _document_loader(self) -> Generator[str, None, None]:
        """
        Load documents from the SQLite database using the configured SQL query. Important: it always yields the element from the first column.
        This must be the column containing documents.

        Yields:
            str: A document as a string from the database.
        """
        conn = sqlite3.connect(self.db_name)
        try:
            res = conn.execute(self.sql_query)
            for row in res:
                yield row[0]
        finally:
            conn.close()

    def update_sql_query(self, sql_query: str) -> None:
        """
        Update the SQL query and refresh the document count.

        Args:
            sql_query (str): New SQL query to fetch documents.
        """
        self.sql_query = sql_query
        self.n_documents = self.__len__()

    def collect_dataframe_from_db(self, sql_query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.

        Args:
            sql_query (str): SQL query to fetch data.

        Returns:
            pd.DataFrame: Resulting data as a DataFrame.
        """
        conn = sqlite3.connect(self.db_name)
        try:
            df = pd.read_sql(sql_query, conn)
        finally:
            conn.close()
        return df

    @classmethod
    def load(cls, file_path: str) -> 'SQLiteCorpus':
        """
        Load a saved SQLiteCorpus object from a file.

        Args:
            file_path (str): Path to the serialized object.

        Returns:
            SQLiteCorpus: The deserialized SQLiteCorpus object.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            db_name=data["db_name"],
            sql_query=data["sql_query"],
            preprocessor=data["preprocessor"],
            identify_phrases=data["identify_phrases"],
            phrases_arguments=data["phrases_arguments"],
            create_dictionary_and_countings=data["create_dictionary_and_countings"],
            dictionary_arguments=data["dictionary_arguments"],
            tfidf_arguments_smart=data["tfidf_arguments_smart"],
            yield_documents=data["yield_documents"],
            train_corpus=False
        )

        # Restore additional attributes
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
            }:
                setattr(instance, key, value)

        return instance

    def __len__(self) -> int:
        """
        Count the number of documents returned by the SQL query.

        Returns:
            int: Number of documents in the corpus.
        """
        return sum(1 for _ in self._document_loader())
    

class ListCorpus(Corpus):
    """
    Corpus implementation for handling a list of in-memory documents.
    """

    def __init__(
        self,
        documents: List[str],
        preprocessor: Callable[[str], List[str]] = simple_preprocess,
        identify_phrases: bool = True,
        phrases_arguments: Optional[Dict[str, Any]] = None,
        create_dictionary_and_countings: bool = True,
        dictionary_arguments: Optional[Dict[str, Union[int, float]]] = None,
        tfidf_arguments_smart: Optional[str] = None,
        yield_documents: bool = False,
        train_corpus: bool = True
    ) -> None:
        """
        Initialize a ListCorpus object with a list of documents.

        Args:
            documents (List[str]): List of text documents.
            preprocessor (Callable[[str], List[str]]): Tokenization and preprocessing function.
            identify_phrases (bool): Whether to detect phrases in the corpus.
            phrases_arguments (Optional[Dict[str, Any]]): Parameters for the Phrases model. See https://radimrehurek.com/gensim/models/phrases.html
            create_dictionary_and_countings (bool): Whether to create a dictionary and apply filters.
            dictionary_arguments (Optional[Dict[str, Union[int, float]]]): Parameters for dictionary filtering. See https://radimrehurek.com/gensim/corpora/dictionary.html and the method filter_extremes
            tfidf_arguments_smart (Optional[str]): Scoring scheme for TF-IDF ("ntn", "nnn", etc.). See https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
            yield_documents (bool): Whether to yield TaggedDocument objects. Only needed when we want to train a Doc2Vec model.
            train_corpus (bool): Whether to initialize models and preprocess the corpus.
        """
        self.documents = documents
        super().__init__(
            preprocessor=preprocessor,
            identify_phrases=identify_phrases,
            phrases_arguments=phrases_arguments,
            create_dictionary_and_countings=create_dictionary_and_countings,
            dictionary_arguments=dictionary_arguments,
            tfidf_arguments_smart=tfidf_arguments_smart,
            yield_documents=yield_documents,
            train_corpus=train_corpus
        )

    def _document_loader(self) -> Generator[str, None, None]:
        """
        Yield documents from the in-memory list.

        Yields:
            str: A document from the list.
        """
        for doc in self.documents:
            yield doc

    @classmethod
    def load(cls, file_path: str) -> 'ListCorpus':
        """
        Load a saved ListCorpus object from a file.

        Args:
            file_path (str): Path to the serialized object.

        Returns:
            ListCorpus: The deserialized ListCorpus object.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            documents=data["documents"],
            preprocessor=data["preprocessor"],
            identify_phrases=data["identify_phrases"],
            phrases_arguments=data["phrases_arguments"],
            create_dictionary_and_countings=data["create_dictionary_and_countings"],
            dictionary_arguments=data["dictionary_arguments"],
            tfidf_arguments_smart=data["tfidf_arguments_smart"],
            yield_documents=data["yield_documents"],
            train_corpus=False
        )

        # Restore additional attributes
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
            }:
                setattr(instance, key, value)

        return instance

    def __len__(self) -> int:
        """
        Get the number of documents in the corpus.

        Returns:
            int: Number of documents in the list.
        """
        return len(self.documents)
    