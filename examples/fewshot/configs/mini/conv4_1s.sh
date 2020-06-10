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