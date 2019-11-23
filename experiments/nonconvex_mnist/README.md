# Nonconvex MNIST

MNIST experiment from p.11 of "Stochastic Variance Reduction for Nonconvex Optimization".


## Results

![](train_loss.png)
![](grad_norm.png)
![](test_error.png)


## Hyperparameters

* Layers: [784, 100, 10]
* L2 Regularization: 1e-3
* Batch Size: 10
* Warmup Epochs: 10
* Inner Epochs: 1
* Outer Epochs: At most 300
* Learning Rate: Tuned on Training Loss (0.05)
* Warmup Learning Rate: Tuned on Training Loss (0.03)


## Commands
```bash
# SVRG Run
train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_0.05.json \
  --output_path experiments/nonconvex_mnist/svrg-0.05.json \
  --dataset MNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate 0.05 \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda

# Plots
python plot.py \
  --run_paths experiments/nonconvex_mnist/svrg-0.05.json \
  --y_top 0.1 \
  --key train_loss \
  --save_path experiments/nonconvex_mnist/train_loss.png
python plot.py \
  --run_paths experiments/nonconvex_mnist/svrg-0.05.json \
  --key grad_norm \
  --log_scale \
  --save_path experiments/nonconvex_mnist/grad_norm.png
python plot.py \
  --run_paths experiments/nonconvex_mnist/svrg-0.05.json \
  --key test_error \
  --y_top 0.03 \
  --save_path experiments/nonconvex_mnist/test_error.png
python plot.py \
  --run_paths mnist_sgd_fixed0001.json mnist_svrg05.json
  --key train_loss
python plot.py \
  --run_paths mnist_sgd_fixed0001.json mnist_svrg05.json
  --key gran_norm
```


## Hyperparameter Tuning

```bash
# Warmup Learning Rate Search
for lr in 0.5 0.25 0.1 0.03 0.01 0.001; do python train.py \
  --seed 12 \
  --optimizer SGD \
  --run_name sgd_$lr.json \
  --output_path experiments/nonconvex_mnist/warmup-$lr.json \
  --dataset MNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --num_epochs 10
done
# Plot Warmup Results
python plot.py \
  --run_paths experiments/nonconvex_mnist/warmup-*.json \
  --key train_loss \
  --save_path experiments/nonconvex_mnist/warmup_results.png

# SVRG Learning Rate Search
for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do python train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_$lr.json \
  --output_path experiments/nonconvex_mnist/svrg-$lr.json \
  --dataset MNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda
done
python plot.py \
  --run_paths experiments/nonconvex_mnist/svrg-*.json \
  --key train_loss \
  --y_top 0.1 \
  --y_bottom 0.05 \
  --save_path experiments/nonconvex_mnist/svrg_results.png
```

### Warmup Learning Rate Results
![Warmup Learning Rate](warmup_results.png "Warmup Learning Rate")

### SVRG Learning Rate Results
![SVRG Learning Rate](svrg_results.png "SVRG Learning Rate")
