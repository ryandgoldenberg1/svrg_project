import argparse
import json
import matplotlib.pyplot as plt


def create_plot(runs, key):
    plt.xlabel('#grad/n')
    plt.ylabel(key)
    for run in runs:
        if run['args']['optimizer'] == 'SGD':
            plot_sgd_run(run=run, key=key)
        elif run['args']['optimizer'] == 'SVRG':
            plot_svrg_run(run=run, key=key)
        else:
            raise ValueError('Unrecognized optimizer: {}'.format(run['args']['optimizer']))
    plt.legend()


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

    inner_epoch_fraction = run['args']['inner_epoch_fraction'] or 1.
    # SVRG paper counts target gradients per inner epoch only in nonconvex case
    grad_epochs_per_inner_epoch = 2 if len(run['args']['layer_sizes']) > 2 else 1
    grad_epochs_per_inner_epoch *= inner_epoch_fraction
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
    if key == 'grad_norm':
        # Squared Grad Norm
        y = [el**2 for el in y]
    print('run:', run_name)
    print('key:', key)
    print('x:', x)
    print('y:', y)
    plt.plot(x, y, label=run_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_paths', nargs='+', required=True)
    parser.add_argument('--key', default='train_loss', choices=['train_loss', 'grad_norm', 'test_error'])
    parser.add_argument('--y_top', type=float)
    parser.add_argument('--y_bottom', type=float)
    parser.add_argument('--log_scale', default=False, action='store_true')
    parser.add_argument('--save_path')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    runs = []
    for run_path in args.run_paths:
        with open(run_path) as f:
            run = json.load(f)
            runs.append(run)
    print('Loaded runs')
    create_plot(runs=runs, key=args.key)
    if args.y_top is not None:
        plt.ylim(top=args.y_top)
        print('Limited top of y axis to:', args.y_top)
    if args.y_bottom is not None:
        plt.ylim(bottom=args.y_bottom)
        print('Limited bottom of y axis to:', args.y_bottom)
    if args.log_scale:
        plt.yscale('log')
        print('Using log scale')
    if args.save_path is not None:
        plt.savefig(args.save_path)
        print('Saved image to:', args.save_path)
    plt.show()


if __name__ == '__main__':
    main()
