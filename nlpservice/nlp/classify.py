import logging
from collections import Counter

import click
from tqdm import tqdm

import numpy as np
from gensim.models import FastText
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Embedding,
                                     GlobalMaxPooling1D, MaxPooling1D)
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from textacy.preprocess import normalize_whitespace

from .models import get_model, gpu_session, nongpu_session
from .prepare import text_tokenize
from .utils import get_lemmatized_kg

logger = logging.getLogger(__name__)


def prepare_corpus(docs, lemmatized_kg):
    """ Get a label for each document. Returns ML compatibile X/y data structs
    """

    unlabeled = 0

    X,  y = [], []

    for doc in tqdm(docs):
        labels = get_doc_labels(doc, lemmatized_kg)

        if not labels:
            unlabeled += 1

            continue

        X.append(doc)
        y.append(labels)

    return X, y


SPECIAL_TOKENS = {'<pad>': 0, '<start>': 1, '<unk>': 2, '<unused>': 3}


def dtm_from_docs(docs, vocab, maxlen):
    """ Transform docs to term matrices, padded to maxlen
    """

    # patch the vocabulary to reserve first positions
    token2id = {v: (i + 4) for i, v in enumerate(vocab)}

    def doc2tokens(doc):
        out = []

        for sent in doc:
            words = sent.split(' ')
            out.extend([token2id.get(w, SPECIAL_TOKENS['<unk>'])
                        for w in words])
            out.append(SPECIAL_TOKENS['<start>'])

        return out

    dtm = [doc2tokens(doc) for doc in docs]
    X = pad_sequences(dtm, maxlen=maxlen, padding='post')

    return X


def create_model(vocab_size, embedding_dimmension, input_length,
                 embedding_vectors, output_size):
    # model parameters
    num_filters = 64
    weight_decay = 1e-4

    model = Sequential()

    model.add(Embedding(
        vocab_size, embedding_dimmension, input_length=input_length,
        weights=[embedding_vectors],
        trainable=False
    ))

    # CNN type of network. Accuracy is above 0.95 if used with lower Dropout
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(output_size, activation='sigmoid'))

    return model


def make_labelencoder(terms):
    model = preprocessing.LabelEncoder()
    model.fit(terms)

    return model


def make_classifier(kvmodel, docs, lemmatized_kg):
    embeddings = kvmodel.wv.vectors
    EMB_DIM = embeddings.shape[1]    # word embedding dimmension
    VOCAB_SIZE = len(embeddings) + len(SPECIAL_TOKENS)
    fill = np.zeros((len(SPECIAL_TOKENS), EMB_DIM))

    MAX_LEN = 300      # Max length of text sequences
    logger.info("Extracting document labels")
    X, y = prepare_corpus(docs, lemmatized_kg)

    # TODO: y labeled output has changed
    # Needs to be fixed

    # one-hot encode labels
    top_labels = list(sorted(lemmatized_kg.keys()))
    sle = make_labelencoder(top_labels)
    y = sle.transform(y)
    y = to_categorical(y, num_classes=len(lemmatized_kg))

    logger.info("Creating DTM training dataset")
    X = dtm_from_docs(X, vocab=kvmodel.wv.index2word, maxlen=MAX_LEN)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=0)

    # training params
    batch_size = 100    # 256
    num_epochs = 80

    logger.info("Creating model")
    emb_vectors = np.vstack((fill, embeddings))
    output_size = len(lemmatized_kg)
    model = create_model(
        vocab_size=VOCAB_SIZE,
        embedding_dimmension=EMB_DIM,
        input_length=MAX_LEN,
        embedding_vectors=emb_vectors,
        output_size=output_size,
    )

    adam = optimizers.Adam(lr=0.001, beta_1=0.9,
                           beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    logger.info('Start training')

    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        callbacks=[
            EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3,
                          verbose=1)],
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
    )

    loss, accuracy = model.evaluate(X_train, y_train, verbose=True)

    return model, loss, accuracy, history


def get_doc_labels(doc, kg):
    found = Counter()

    # TODO: should also stem the doc sentences
    # or parse with Spacy and do word match

    for label, stem_pairs in kg.items():

        for sent in doc:
            for stems in stem_pairs:
                if all([stem in sent for stem in stems]):
                    found[label] += 1

    top = found.most_common()

    return top or []
    #
    # if top:
    #     return top[0][0]
    #
    # return []


def read_corpus(path):
    """ Returns a list of documents. A doc is a list of sentences.

    A sentence is space separated tokens (words)
    """

    logger.info('Loading corpus')
    docs = []
    doc = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()

            if not line:
                docs.append(doc)
                doc = []
            else:
                doc.append(line)

    return docs


def stream_corpus(path):
    """ Yields documents. A doc is a list of sentences.

    A sentence is space separated tokens (words)
    """

    doc = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()

            if not line:
                yield doc
                doc = []
            else:
                doc.append(line)


@click.command()
@click.argument('output')
@click.argument('ftpath')
@click.argument('corpus')
@click.option('--kg-url',
              default='http://app:8880/api/knowledge-graph/dump_all/',
              help='KnowledgeGraph dump location')
@click.option('--cpu/--gpu',
              default=True,
              help='Use CPU for training, if no GPU is available')
def main(output, ftpath, corpus, kg_url, cpu):
    """ Train a Classification model

    :param output: output path for the TF model
    :param ftpath: path for Fasttext word embedding model
    :param corpus: path to a text file of documents. One sentence per line.
                    Separate documents with an empty line
    :
    """
    logger.setLevel(logging.DEBUG)
    docs = read_corpus(corpus)

    logger.info("Loading fasttext model")
    ft_model = FastText.load(ftpath)
    kg = get_lemmatized_kg(kg_url)

    if cpu:
        sess = nongpu_session()
    else:
        sess = gpu_session()  # non
    with sess.as_default():
        k_model, loss, accuracy, history = make_classifier(
            ft_model, docs, kg
        )

        logger.info("Model trained, saving")
        save_model(k_model, output, overwrite=True, include_optimizer=True)

    return output


@click.command()
@click.argument('corpus')
@click.argument('output')
@click.option('--kg-url',
              default='http://app:8880/api/knowledge-graph/dump_all/',
              help='KnowledgeGraph dump location')
@click.option('--test-size', default=0.3, help='Split ratio', type=float)
def label(corpus, output, kg_url, test_size):
    """ Generate fasttext compatible text files
    """

    docs = read_corpus(corpus)
    kg = get_lemmatized_kg(kg_url)
    X, y = prepare_corpus(docs, kg)
    yy = []

    for tl in y:
        ls = [x[0].lower().replace(' ', '_') for x in tl]
        yy.append(ls)
    y = yy

    X = [normalize_whitespace(('</s>'.join(sents)).replace('dignr', ''))
         for sents in X]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=0)
    train_path = output + '-train'
    test_path = output + '-test'

    with open(train_path, 'w') as f:
        for labels, text in zip(y_train, X_train):
            ls = ' '.join(['__label__{}'.format(l) for l in labels])
            line = "{} {}".format(ls, text)
            f.write(line)
            f.write('\n')

    with open(test_path, 'w') as f:
        for labels, text in zip(y_test, X_test):
            ls = ' '.join(['__label__{}'.format(l) for l in labels])
            line = "{} {}".format(ls, text)
            f.write(line)
            f.write('\n')

    logger.info("Wrote train file: %s", train_path)
    logger.info("Wrote test file: %s", test_path)


def _predict(text, model, label_encoder, vocab, maxlen):
    # transform the document to a list of sentences (list of tokens)
    doc = text_tokenize(text)

    # compatibility with dtm_from_docs
    doc = [' '.join(words) for words in doc]
    X = dtm_from_docs([doc], vocab, maxlen)

    return model.predict(X)


def predict_classes(text, model_name):
    """ Make class predictions for text
    """

    suite = get_model(model_name)

    model = suite['model']
    label_encoder = suite['labels']
    vocab = suite['vocab']

    maxlen = model.inputs[0].get_shape()[1].value

    k = _predict(text, model, label_encoder, vocab, maxlen)

    pairs = zip(map(str, k.ravel()),
                label_encoder.classes_)

    return list(pairs)


def load_classifier_model(loader):
    """ Generic loader for classification models.

    :param loader: a callable that returns model path, word embeddings path
                   (for vocabulary) and the labels (classification targets)
    """

    model_path, ft_model_path, labels = loader()

    session = nongpu_session()

    with session.as_default():
        model = load_model(model_path)

    kv_model = FastText.load(ft_model_path)
    vocab = kv_model.wv.index2word
    label_encoder = make_labelencoder(labels)

    return {
        'kv_model': kv_model,
        'labels': label_encoder,
        'model': model,
        'vocab': vocab,
    }


def kg_classify_settings(config):
    """ A classifier that uses the top labels KnowledgeGraph as classes
    """

    settings = config.get_settings()

    kg_model_path = settings['nlp.kg_model_path']
    ft_model_path = settings['nlp.kg_ft_path']
    kg_url = settings['nlp.kg_url']

    kg = get_lemmatized_kg(kg_url)
    labels = list(sorted(kg.keys()))

    return kg_model_path, ft_model_path, labels


def make_predict(config):
    pass
