import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as f:
    README = f.read()


install_requires = [
    'gunicorn',

    'cornice',
    'waitress',
    'tensorflow-hub',
    'colander',
    'scipy',
    'gensim',

    # text cleanup utilities
    'textacy',
    'ftfy',
    'syntok',
    'truecase',

    # command line utils
    'click',
]

testing_requires = [
    'pytest',
    'pytest-cov'
]


setup(name='nlpservice',
      version=0.1,
      description='NLP Services via REST',
      long_description=README,
      classifiers=[
          "Programming Language :: Python",
          "Framework :: Pylons",
          "Topic :: Internet :: WWW/HTTP",
          "Topic :: Internet :: WWW/HTTP :: WSGI :: Application"
      ],
      keywords="web services",
      author='',
      author_email='',
      url='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      extras_require={
          'testing': testing_requires,
      },
      entry_points="""\
      [paste.app_factory]
      main=nlpservice:main
      [console_scripts]
      prepare=nlpservice.nlp.prepare:main
      kv=nlpservice.nlp.fasttext:main
      train=nlpservice.nlp.classify:main
      label=nlpservice.nlp.classify:label
      """,
      # retrain=nlpservice.nlp.classify:retrain
      paster_plugins=['pyramid'])
