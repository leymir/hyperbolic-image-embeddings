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
  --dim 1024