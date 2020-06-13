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