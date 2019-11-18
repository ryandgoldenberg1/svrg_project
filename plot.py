import argparse
import json
import matplotlib.pyplot as plt



def create_plot(runs, key):
    plt.xlabel('#grad/n')
    plt.ylabel(key)
    for run in runs:
        if run['script'] == 'sgd.py':
            plot_sgd_run(run=run, key=key)
        elif run['script'] == 'svrg.py':
            plot_svrg_run(run=run, key=key)
        else:
            raise ValueError('Unrecognized script: {}'.format(run['script']))
    plt.legend()
    plt.show()


def plot_sgd_run(run, key):
    run_name = run['args']['run_name']
    metrics = run['metrics']
    x = [metric['epoch'] for metric in metrics]
    y = [metric[key] for metric in metrics]
    plt.plot(x, y, label=run_name)
    print('run:', run_name)
    print('key:', key)
    print('x:', x)
    print('y:', y)


def plot_svrg_run(run, key):
    run_name = run['args']['run_name']
    metrics = run['metrics']
    # SVRG paper counts target gradients per inner epoch only in nonconvex case
    grad_epochs_per_inner_epoch = 2 if len(run['args']['layer_sizes']) > 2 else 1
    grad_epochs_per_outer_epoch = 1 + run['args']['num_inner_epochs'] * grad_epochs_per_inner_epoch
    x = []
    for metric in metrics:
        warmup_epoch = metric.get('warmup_epoch') or run['args']['num_warmup_epochs']
        outer_epoch = metric.get('outer_epoch') or 0
        inner_epoch = metric.get('inner_epoch') or 0

        grad_epoch = warmup_epoch
        if outer_epoch > 0:
            grad_epoch += (outer_epoch - 1) * grad_epochs_per_outer_epoch
            grad_epoch += 1
        if inner_epoch > 0:
            grad_epoch += inner_epoch * grad_epochs_per_inner_epoch
        x.append(grad_epoch)
    y = [metric[key] for metric in metrics]
    print('run:', run_name)
    print('key:', key)
    print('x:', x)
    print('y:', y)
    plt.plot(x, y, label=run_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_paths', nargs='+', required=True)
    parser.add_argument('--key', default='train_loss')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    runs = []
    for run_path in args.run_paths:
        with open(run_path) as f:
            run = json.load(f)
            runs.append(run)
    print('Loaded runs')
    create_plot(runs=runs, key=args.key)


if __name__ == '__main__':
    main()
