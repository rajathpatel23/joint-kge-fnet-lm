#!/usr/bin/env sh

set -o errexit
set -o nounset

echo "Downloading Corpus"
mkdir data/download_dataset/
wget https://drive.google.com/drive/folders/1UYJ9GkuaDGNEcgvVi85KWJ-UnECQ1Vdb?usp=sharing -P data/download_dataset/

echo "Downloading word embeddings...."
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
mv glove.840B.300d.txt data/download_dataset/

echo "preprocessing dataset - WN11"
python data/WN11/preprocessing_file_1.py data/download_dataset/WN11/ data/download_dataset/ WN11
python data/WN11/data_collector.py data/download_dataset/WN11/ train Sample_data_head_24.pkl 2 4 1

echo "preprocessing dataset - FB13"
python data/fb13_7/preprocessing_file_1.py data/download_dataset/fb13/ data/download_dataset/ FB13
python data/fb13_7/data_collector.py data/download_dataset/fb13/ train Sample_data_head_24.pkl 2 4 1

echo "preprocessing data - WN18RR"
python data/WN18RR/preprocessing.py data/download_dataset/WN18RR/ data/download_dataset/ WN18RR
python data/WN18RR/data_collector.py data/download_dataset/WN18RR/ train Sample_data_head_24.pkl 2 4 1
python data/WN18RR/data_collector.py data/download_dataset/WN18RR/ valid dev_vec_dict.pkl 2 2 1
python data/WN18RR/data_collector.py data/download_dataset/WN18RR/ test test_vec_dict.pkl 2 2 1

echo "preprocessing data - FB15k-237"
python data/FB15K-237/preprocessing.py data/download_dataset/FB15K-237/ data/download_dataset/ FB15K-237 data/download_dataset/FB15K-237/entity2wikidata.json
python data/FB15K-237/data_collector.py data/download_dataset/FB15K-237/ Sample_data_head_24.pkl 2 4 1 data/download_dataset/FB15K-237/relation_2_tail.pkl train.txt train data/download_dataset/FB15K-237/entity2wikidata.json
python data/FB15K-237/data_collector.py data/download_dataset/FB15K-237/ dev_vec_dict.pkl 2 2 1 data/download_dataset/FB15K-237/relation_2_tail_dev.pkl valid.txt dev data/download_dataset/FB15K-237/entity2wikidata.json
python data/FB15K-237/data_collector.py data/download_dataset/FB15K-237/ test_vec_dict.pkl 2 2 1 data/download_dataset/FB15K-237/relation_2_tail_test.pkl test.txt test data/download_dataset/FB15K-237/entity2wikidata.json

echo "preprocessing data - FB15K"
python data/FB15K/preprocessing.py data/download_dataset/FB15K/ data/download_dataset/ FB15K data/download_dataset/FB15K/entity2wikidata.json
python data/FB15K/data_collector.py data/download_dataset/FB15K/ Sample_data_head_24.pkl 2 4 1 data/download_dataset/FB15K/relation_2_tail.pkl train.txt train data/download_dataset/FB15K/entity2wikidata.json
python data/FB15K/data_collector.py data/download_dataset/FB15K/ dev_vec_dict.pkl 2 2 1 data/download_dataset/FB15K/relation_2_tail_dev.pkl valid.txt dev data/download_dataset/FB15K/entity2wikidata.json
python data/FB15K/data_collector.py data/download_dataset/FB15K/ test_vec_dict.pkl 2 2 1 data/download_dataset/FB15K/relation_2_tail_test.pkl test.txt test data/download_dataset/FB15K/entity2wikidata.json





