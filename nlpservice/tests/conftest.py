import json

import pytest
from pkg_resources import resource_filename


@pytest.fixture
def kg():
    from nlpservice.nlp.classify import flatten_knowledge_graph

    fpath = resource_filename('nlpservice', 'tests/fixtures/kg.json')
    with open(fpath) as f:
        j = json.loads(f.read())

    return flatten_knowledge_graph(j)


@pytest.fixture
def lemmatized_kg():
    from nlpservice.nlp.classify import flatten_knowledge_graph
    from nlpservice.nlp.utils import lemmatize_kg_terms

    fpath = resource_filename('nlpservice', 'tests/fixtures/kg.json')
    with open(fpath) as f:
        kg = json.loads(f.read())

    f_kg = flatten_knowledge_graph(kg)
    l_kg = {}

    for b, terms in f_kg.items():
        l_kg[b] = lemmatize_kg_terms(terms)

    return l_kg


@pytest.fixture
def corpus():
    from nlpservice.nlp.classify import read_corpus
    fpath = resource_filename('nlpservice', 'tests/fixtures/corpus.txt')

    return read_corpus(fpath)


@pytest.fixture
def es_docs():

    fpath = resource_filename('nlpservice', 'tests/fixtures/dump.json')
    with open(fpath) as f:
        j = json.loads(f.read())

    return j


@pytest.fixture
def ftmodel():
    from gensim.models import FastText
    fpath = resource_filename('nlpservice', 'tests/fixtures/corpus-ft')

    return FastText.load(fpath)
