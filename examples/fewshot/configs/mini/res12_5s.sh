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