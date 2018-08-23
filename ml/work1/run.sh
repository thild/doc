#!/bin/bash

#Dataset from http://ai.stanford.edu/%7Eamaas/data/sentiment/ -> http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz
if [ ! -f 'dataset_processed' ]; then
    echo 'Creating data directory...'
    mkdir -p data
    mkdir -p data/tmp
    if [ ! -f 'data/aclImdb_v1.tar.gz' ]; then
        echo 'Downloading IMDB dataset...'
        curl -o data/aclImdb_v1.tar.gz -LO http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz
    fi
    if [ ! -d 'data/aclImdb' ]; then
        echo 'Extracting IMDB dataset...'
        tar -xvzf data/aclImdb_v1.tar.gz -C data
    fi
    #curl -o data/nounlist.txt -LO http://www.desiquintans.com/downloads/nounlist/nounlist.txt
    echo 'Pre-processing datasets...'
    awk '{gsub("<[^>]*>", ""); gsub("[^a-zA-Z''\\s]|\\s\\w{1,2}\\s", " ")}1' data/aclImdb/train/unsup/{0..4}_0.txt > data/tmp/test.txt
    awk '{gsub("<[^>]*>", ""); gsub("[^a-zA-Z''\\s]|\\s\\w{1,2}\\s", " ")}1' data/aclImdb/test/neg/*.txt > data/tmp/test_neg.txt
    awk '{gsub("<[^>]*>", ""); gsub("[^a-zA-Z''\\s]|\\s\\w{1,2}\\s", " ")}1' data/aclImdb/test/pos/*.txt > data/tmp/test_pos.txt
    awk '{gsub("<[^>]*>", ""); gsub("[^a-zA-Z''\\s]|\\s\\w{1,2}\\s", " ")}1' data/aclImdb/train/neg/*.txt > data/tmp/train_neg.txt
    awk '{gsub("<[^>]*>", ""); gsub("[^a-zA-Z''\\s]|\\s\\w{1,2}\\s", " ")}1' data/aclImdb/train/pos/*.txt > data/tmp/train_pos.txt
    awk '{gsub("<[^>]*>", ""); gsub("[^a-zA-Z''\\s]|\\s\\w{1,2}\\s", " ")}1' data/aclImdb/train/unsup/{0..24999}_0.txt > data/tmp/test_unsup1.txt
    awk '{gsub("<[^>]*>", ""); gsub("[^a-zA-Z''\\s]|\\s\\w{1,2}\\s", " ")}1' data/aclImdb/train/unsup/{25000..49999}_0.txt > data/tmp/test_unsup2.txt
    python3 process_files.py
    touch dataset_processed
fi

python3 run.py