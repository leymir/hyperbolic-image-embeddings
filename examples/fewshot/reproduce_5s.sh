# FIVE SHOT FIVE WAY

# CUB 5s5w (conv4) 0.8253 +- 0.0014
python train_protonet.py \
  --dataset CUB \
  --shot 5 \
  --lr 0.001 \
  --step 40 \
  --gamma 0.8 \
  --c 0.01 \
  --model convnet \
  --hyperbolic \
  --not-riemannian \
  --dim 1600

# MiniImageNet 5s5w (conv4)  0.7269 +- 0.0014
python train_protonet.py \
  --dataset MiniImageNet \
  --way 20 \
  --shot 5 \
  --lr 0.005 \
  --step 60 \
  --gamma 0.8 \
  --c 0.005 \
  --model convnet \
  --hyperbolic \
  --not-riemannian \
  --dim 1600

# MiniImageNet 5s5w (resnet18) 0.7684 +/- 0.0014
python train_protonet.py \
  --dataset MiniImageNet \
  --way 20 \
  --shot 5 \
  --lr 0.001 \
  --step 40 \
  --gamma 0.5 \
  --c 0.005 \
  --model resnet18 \
  --hyperbolic \
  --not-riemannian \
  --dim 512

# MiniImageNet 5s5w (resnet12) 0.7819 +- 0.0014
python train_protonet.py \
  --dataset MiniImageNet \
  --way 20 \
  --shot 5 \
  --lr 0.005 \
  --step 50 \
  --gamma 0.8 \
  --c 0.005 \
  --model resnet12 \
  --hyperbolic \
  --not-riemannian \
  --dim 512



