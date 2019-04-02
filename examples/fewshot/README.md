## Few shot classification on MiniImageNet and Caltech-UCSD Birds-200-2011

Here we provide the code necessary to replicate our results on few-shot classification datasets.

### Dataset preparation
Simply run 
`bash get_birds.sh` and `bash get_mini.sh` to download and unpack the corresponding datasets.

### Training

To train the network run `python train_protonet.py`. Some of the available arguments are:

- `--shot` for n-shot training (default 1)
- `--query` query size (default 15)
- `--way` k--way during training (default 5)
- `--validation_way` k-way during validation (default 5)
- `--dataset` which dataset to train on, choices are CUB and MiniImageNet
- `--hyperbolic` whether to use the hyperbolic network (default False)
- `--c` curvature parameter of the Poincare ball (default 1.0)
- `--dim` dimensionality of the embeddings (default 64)
- `--train_c` whether to train the curvature parameter (for the hyperbolic network)
- `--train_x` whether to train the origin point of the exponential map (for the hyperbolic network)

###

Example scripts

- `python train_protonet.py --gpu 0 --hyperbolic --dataset CUB --dim 512 --lr 0.001 --c 0.05 --gamma 0.7 --step_size 20`
- `python train_protonet.py --gpu 0 --hyperbolic --dataset MiniImageNet --dim 1024 --lr 0.001 --c 0.05 --gamma 0.2 
--step_size 10`