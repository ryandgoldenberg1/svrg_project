Following command shown how to run the experiment and generate the corresponding figure

# Reproducibility of paper result

# Critical evaluatin result

# Hyperparameteer tuning result
## SVRG
Note: you can change the `--dataset` to CIFAR10 or STl to get the corresponding plot
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
