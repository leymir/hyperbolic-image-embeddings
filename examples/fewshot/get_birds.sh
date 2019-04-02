cd data/cub
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
tar -xzvf images.tgz
rm images.tgz
cd ../..
mv data/cub/images/*/*.jpg data/cub/images
