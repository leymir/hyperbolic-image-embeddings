#!/usr/bin/env bash
cd data/cub
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzvf CUB_200_2011.tgz
rm CUB_200_2011.tgz
rm attributes.txt
mkdir images
mv CUB_200_2011/images/*/*.jpg images
rm -r CUB_200_2011
cd ../..
