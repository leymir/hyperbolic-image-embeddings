# ONE SHOT FIVE WAY

# CUB 1s5w (conv4) 0.6402 +- 0.002
python train_protonet.py \
  --dataset CUB \
  --shot 1 \
  --lr 0.001 \
  --step 50 \
  --gamma 0.8 \
  --c 0.05 \
  --model convnet \
  --hyperbolic \
  --not-riemannian \
  --dim 1600

# MiniImageNet 1s5w (conv4)  0.5443 +- 0.002
python train_protonet.py \
  --dataset MiniImageNet \
  --way 30 \
  --shot 1 \
  --lr 0.005 \
  --step 80 \
  --gamma 0.5 \
  --c 0.01 \
  --model convnet \
  --hyperbolic \
  --not-riemannian \
  --dim 1600

# MiniImageNet 1s5w (resnet18) 0.5947 +/- 0.002
python train_protonet.py \
  --dataset MiniImageNet \
  --way 30 \
  --shot 1 \
  --lr 0.001 \
  --step 80 \
  --gamma 0.5 \
  --c 0.01 \
  --model resnet18 \
  --hyperbolic \
  --not-riemannian \
  --dim 512

# MiniImageNet 1s5w (resnet12) 0.6160 +- 0.002
python train_protonet.py \
  --dataset MiniImageNet \
  --way 30 \
  --shot 1 \
  --lr 0.001 \
  --step 40 \
  --gamma 0.8 \
  --c 0.01 \
  --model resnet12 \
  --hyperbolic \
  --not-riemannian \
  --dim 512
