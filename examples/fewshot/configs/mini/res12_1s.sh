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