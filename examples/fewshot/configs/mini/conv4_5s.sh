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