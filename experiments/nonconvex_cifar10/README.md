# Nonconvex CIFAR10

CIFAR10 experiment from p.11 of "Stochastic Variance Reduction for Nonconvex Optimization".


## Results

![](train_loss.png)
![](grad_norm.png)
![](test_error.png)


## Hyperparameters

* Layers: [3072, 100, 10]
* L2 Regularization: 1e-3
* Batch Size: 10
* Warmup Epochs: 10
* Inner Epochs: 1
* Outer Epochs: At most 300
* Learning Rate: Tuned on Training Loss (0.001)
* Warmup Learning Rate: Tuned on Training Loss (0.01)

## Commands

```bash
# SVRG Run
python train.py \
  --seed 83 \
  --optimizer SVRG \
  --run_name svrg_0.001.json \
  --output_path experiments/nonconvex_cifar10/svrg-0.001.json \
  --dataset CIFAR10 \
  --layer_sizes 3072 100 10 \
  --batch_size 10 \
  --learning_rate 0.001 \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.01 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda

# Plots
python plot.py \
  --run_paths experiments/nonconvex_cifar10/svrg-0.001.json \
  --key train_loss \
  --save_path experiments/nonconvex_cifar10/train_loss.png
python plot.py \
  --run_paths experiments/nonconvex_cifar10/svrg-0.001.json \
  --key grad_norm \
  --log_scale \
  --save_path experiments/nonconvex_cifar10/grad_norm.png
python plot.py \
  --run_paths experiments/nonconvex_cifar10/svrg-0.001.json \
  --key test_error \
  --save_path experiments/nonconvex_cifar10/test_error.png
```

## Hyperparameter Tuning

```bash
# Warmup Learning Rate Search
for lr in 0.5 0.25 0.1 0.03 0.01 0.001; do python train.py \
  --seed 12 \
  --optimizer SGD \
  --run_name sgd_$lr.json \
  --output_path experiments/nonconvex_cifar10/warmup-$lr.json \
  --dataset CIFAR10 \
  --layer_sizes 3072 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --num_epochs 10
done
python plot.py \
  --run_paths experiments/nonconvex_cifar10/warmup-*.json \
  --key train_loss \
  --save_path experiments/nonconvex_cifar10/warmup_results.png

# SVRG Learning Rate Search
for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do python train.py \
  --seed 83 \
  --optimizer SVRG \
  --run_name svrg_$lr.json \
  --output_path experiments/nonconvex_cifar10/svrg-$lr.json \
  --dataset CIFAR10 \
  --layer_sizes 3072 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.01 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda
done
python plot.py \
  --run_paths experiments/nonconvex_cifar10/svrg-*.json \
  --key train_loss \
  --save_path experiments/nonconvex_cifar10/svrg_results.png
```

### Warmup Learning Rate Results
![Warmup Learning Rate](warmup_results.png "Warmup Learning Rate")

### SVRG Learning Rate Results
![SVRG Learning Rate](svrg_results.png "SVRG Learning Rate")
