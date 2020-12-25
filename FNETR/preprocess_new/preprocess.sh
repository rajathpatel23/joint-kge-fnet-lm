#!/usr/bin/env sh

set -o errexit
set -o nounset

echo "Downloading Corpus"
wget http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip
unzip corpus.zip
rm corpus.zip

echo "Downloading word embeddings...."
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
mv glove.840B.300d.txt resource_1/

echo "Preprocessing (creating ids for words, features, and labels, creating datasets)"

echo "OntoNotes"
python create_X2id.py corpus/OntoNotes/all.txt resource_1/OntoNotes/word2id_gillick.txt resource_1/OntoNotes/feature2id_gillick.txt resource_1/OntoNotes/label2id_gillick.txt
python create_dicts.py resource_1/OntoNotes/word2id_gillick.txt resource_1/OntoNotes/feature2id_gillick.txt  resource_1/OntoNotes/label2id_gillick.txt  resource_1/glove.840B.300d.txt resource_1/OntoNotes/dicts_gillick.pkl
python create_dataset.py resource_1/OntoNotes/dicts_gillick.pkl corpus/OntoNotes/train.txt resource_1/OntoNotes/train_gillick.pkl
python create_dataset.py resource_1/OntoNotes/dicts_gillick.pkl corpus/OntoNotes/dev.txt resource_1/OntoNotes/dev_gillick.pkl
python create_dataset.py resource_1/OntoNotes/dicts_gillick.pkl corpus/OntoNotes/test.txt resource_1/OntoNotes/test_gillick.pkl


echo "Wiki"
python create_X2id.py corpus/Wiki/all.txt resource_1/Wiki/word2id_figer.txt resource_1/Wiki/feature2id_figer.txt resource_1/Wiki/label2id_figer.txt
python create_dicts.py resource_1/Wiki/word2id_figer.txt resource_1/Wiki/feature2id_figer.txt resource_1/Wiki/label2id_figer.txt  resource_1/glove.840B.300d.txt resource_1/Wiki/dicts_figer.pkl
python create_dataset.py resource_1/Wiki/dicts_figer.pkl corpus/Wiki/train.txt resource_1/Wiki/train_figer.pkl
python create_dataset.py resource_1/Wiki/dicts_figer.pkl corpus/Wiki/dev.txt resource_1/Wiki/dev_figer.pkl
python create_dataset.py resource_1/Wiki/dicts_figer.pkl corpus/Wiki/test.txt resource_1/Wiki/test_figer.pkl
