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
  --dim 1024