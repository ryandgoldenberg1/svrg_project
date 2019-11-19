STL10 experiment from p.11 of "Stochastic Variance Reduction for Nonconvex Optimization".

## Parameters

* Layers: [27648, 100, 10]
* L2 Regularization: 1e-2
* Batch Size: 10
* Warmup Epochs: 20
* Inner Epochs: 1
* Outer Epochs: At most 300
* Learning Rate: Tuned on Training Loss
* Warmup Learning Rate: Tuned on Training Loss (0.001)


## Commands


## Tuning
```bash
# Warmup Learning Rate Search
for lr in 0.5 0.25 0.1 0.03 0.01 0.001; do python train.py \
  --seed 12 \
  --optimizer SGD \
  --run_name sgd_$lr.json \
  --output_path experiments/nonconvex_stl10/warmup-$lr.json \
  --dataset STL10 \
  --layer_sizes 27648 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.01 \
  --num_epochs 20
done
python plot.py \
  --run_paths experiments/nonconvex_stl10/warmup-*.json \
  --key train_loss \
  --save_path experiments/nonconvex_stl10/warmup_results.png

# SVRG Learning Rate Search
for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do python train.py \
  --seed 77 \
  --optimizer SVRG \
  --run_name svrg_$lr.json \
  --output_path experiments/nonconvex_stl10/svrg-$lr.json \
  --dataset STL10 \
  --layer_sizes 27648 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.01 \
  --warmup_learning_rate 0.001 \
  --num_warmup_epochs 20 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda
done
```
