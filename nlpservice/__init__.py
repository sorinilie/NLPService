""" Main pyramid entry point, console utils
"""

import os

from nlpservice.nlp.classify import load_classifier_model
from nlpservice.nlp.fasttext import load_kv_model
from nlpservice.nlp.models import MODELS
from pyramid.config import Configurator
from pyramid.util import DottedNameResolver


def load_models(config, prefix, model_loader):

    settings = config.get_settings()

    for k, v in settings.items():
        if not k.startswith(prefix):
            continue

        name = k.rsplit('.', 1)[1]

        for dottedname in v.split():
            func = DottedNameResolver().resolve(dottedname)

            def loader():
                return func(config)

            MODELS[name] = (loader, model_loader)


def main(global_config, **settings):
    config = Configurator(settings=settings)
    config.include("cornice")
    config.scan("nlpservice.views")

    settings = config.get_settings()
    cache_path = settings['nlp.tf_model_cache_path'].strip()
    os.environ['TFHUB_CACHE_DIR'] = cache_path

    load_models(config, 'nlp.classifiers', load_classifier_model)
    load_models(config, 'nlp.keyedvectors', load_kv_model)

    return config.make_wsgi_app()
