import os
import re
import nltk
import pickle
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('wordnet')

class Indexer:
    db_file = "./ir.idx"  # This is the index file you will create and manager for indexing

    def __init__(self):
        self.tok2idx = {}                       # map (token to id)
        self.idx2tok = {}                       # map (id to token)
        self.postings_lists = {}                # postings for each word
        self.docs = []                            # encoded document list
        self.raw_ds = []                        # raw documents for search results
        self.corpus_stats = { 'avgdl': 0.0, 'vocab': set() }        # any corpus-level statistics
        self.stopwords = set(stopwords.words('english'))

        if os.path.exists(self.db_file):
            with open(self.db_file, "rb") as file:
                object = pickle.load(file)
                self.tok2idx = object.tok2idx
                self.idx2tok = object.idx2tok
                self.postings_lists = object.postings_lists
                self.docs = object.docs
                self.raw_ds = object.raw_ds
                self.corpus_stats = object.corpus_stats
                self.stopwords = object.stopwords
        else:
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']
            self.clean_text(self.raw_ds)
            self.create_postings_lists()

    def clean_text(self, lst_text, query=False):
        tokenizer = WhitespaceTokenizer()
        lemmatizer = WordNetLemmatizer()

        if query:
            _query = re.sub(r"[^\w\s()-]", "", lst_text)
            _query = re.sub(r"[()]", " ", _query).lower()
            tokens = [lemmatizer.lemmatize(token) for token in tokenizer.tokenize(_query) if token not in self.stopwords]
            _extra_tokens = list()
            for token in tokens:
                if "-" in token:
                    _extra_tokens.extend(token.split("-"))
            tokens.extend(_extra_tokens)
            return tokens

        for idx, article in enumerate(lst_text):
            # removes punctuation and makes lower case
            _article = re.sub(r"[^\w\s()-]", "", article)
            _article = re.sub(r"[()]", " ", _article).lower()
            # lemmatize and then tokenize by whitespace (removes stopwords)
            tokens = [lemmatizer.lemmatize(token) for token in tokenizer.tokenize(_article) if token not in self.stopwords]
            # tokens = [token for token in tokenizer.tokenize(_article) if token not in self.stopwords]
            _extra_tokens = list()
            for token in tokens:
                if "-" in token:
                    _extra_tokens.extend(token.split("-"))
            tokens.extend(_extra_tokens)
            self.docs.append(tokens)
            self.corpus_stats['vocab'].update(tokens)
        self.idx2tok = {k: v for k, v in enumerate(self.corpus_stats['vocab'])}
        self.tok2idx = {v: k for k, v in self.idx2tok.items()}

    def create_postings_lists(self):
        doc_len = 0
        # adds article to postings
        for idx, article in enumerate(self.docs):
            doc_len += len(article)
            for token in article:
                if token not in self.postings_lists:
                    self.postings_lists[token] = {idx}
                self.postings_lists[token].add(idx)
        # calculate avgdl
        self.corpus_stats["avgdl"] = round(doc_len / len(self.docs), 2)

        with open(self.db_file, "wb") as filehandler:
            pickle.dump(self, filehandler)
