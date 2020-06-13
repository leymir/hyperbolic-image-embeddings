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