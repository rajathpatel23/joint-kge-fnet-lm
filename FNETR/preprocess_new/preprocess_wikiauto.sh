#!/usr/bin/env sh

echo "Downloading WikiAuto data from KNET github link"
wget https://github.com/thunlp/KNET/tree/master/data resource_1/WikiAuto/

cd ./resource_1/WikiAuto/

gunzip *.gz

cd ../../
python numpy_to_dict.py train resource_1/WikiAuto/
python numpy_to_dict.py test resource_1/WikiAuto/
python numpy_to_dict.py valid resource_1/WikiAuto/
python numpy_to_dict.py manual resource_1/WikiAuto/

python token_embedding.py resource_1/glove.840B.300d.txt

python preprocess_data.py train
python preprocess_data.py test
python preprocess_data.py valid
python preprocess_data.py manual
