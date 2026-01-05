#!/usr/bin/env python3
"""
Plot training log metrics with publication-quality output.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import argparse


# Publication-quality settings
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5


def is_lower_better(column_name):
    """Determine if lower values are better for a metric."""
    lower_better_keywords = ['loss', 'mae', 'mse', 'rmse', 'kl', 'time']
    return any(keyword in column_name.lower() for keyword in lower_better_keywords)


def get_metric_display_name(column_name):
    """Convert column name to display-friendly format."""
    # Replace underscores with spaces and capitalize
    name = column_name.replace('_', ' ').title()

    # Special cases for acronyms
    replacements = {
        'Ssim': 'SSIM',
        'Psnr': 'PSNR',
        'Mae': 'MAE',
        'Mse': 'MSE',
        'Rmse': 'RMSE',
        'Kl': 'KL',
        'R2': 'R²',
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name


def plot_training_logs(csv_path, output_dir=None, checkpoint_interval=20):
    """
    Plot all metrics from training logs with publication quality.

    Args:
        csv_path: Path to CSV file with training logs
        output_dir: Directory to save plots (default: same as CSV file)
        checkpoint_interval: Interval at which checkpoints are saved (default: 20)
    """
    # Read the CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  - Found {len(df)} rows, epochs {df['epoch'].min()}-{df['epoch'].max()}")

    # Set up output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent / "plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)
    print(f"  - Saving plots to: {output_dir}")

    # Filter to checkpoint epochs
    checkpoint_epochs = df[df['epoch'] % checkpoint_interval == 0].copy()
    print(f"  - Found {len(checkpoint_epochs)} checkpoint epochs (every {checkpoint_interval} epochs)")

    # Organize metrics by category
    train_metrics = [col for col in df.columns if col.startswith('train_') and col != 'train_loss']
    val_loss_metrics = [col for col in df.columns if col.startswith('val_') and 'loss' in col]
    val_quality_metrics = [col for col in df.columns if col.startswith('val_') and
                          any(m in col for m in ['psnr', 'ssim', 'r2'])]
    val_error_metrics = [col for col in df.columns if col.startswith('val_') and
                        any(m in col for m in ['mae', 'mse', 'rmse'])]
    other_metrics = [col for col in df.columns if col not in ['epoch'] and
                    col not in train_metrics and col not in val_loss_metrics and
                    col not in val_quality_metrics and col not in val_error_metrics and
                    col not in ['learning_rate', 'epoch_time', 'validation_time']]

    # Plot combined training metrics
    if 'train_loss' in df.columns or train_metrics:
        plot_combined_metrics(df, checkpoint_epochs, ['train_loss'] + train_metrics,
                            'Training Metrics', output_dir, 'training_metrics.png')

    # Plot combined validation loss metrics
    if val_loss_metrics:
        plot_combined_metrics(df, checkpoint_epochs, val_loss_metrics,
                            'Validation Loss Metrics', output_dir, 'validation_loss_metrics.png')

    # Plot validation quality metrics
    if val_quality_metrics:
        plot_combined_metrics(df, checkpoint_epochs, val_quality_metrics,
                            'Validation Quality Metrics', output_dir, 'validation_quality_metrics.png')

    # Plot validation error metrics
    if val_error_metrics:
        plot_combined_metrics(df, checkpoint_epochs, val_error_metrics,
                            'Validation Error Metrics', output_dir, 'validation_error_metrics.png')

    # Plot training vs validation comparison
    plot_training_validation_comparison(df, checkpoint_epochs, output_dir)

    # Plot learning rate and time metrics
    plot_auxiliary_metrics(df, output_dir)

    print(f"\n✓ All plots saved to: {output_dir}")


def plot_combined_metrics(df, checkpoint_df, metrics, title, output_dir, filename):
    """Plot multiple related metrics in subplots."""
    # Filter out metrics that are all NaN or zero
    valid_metrics = [m for m in metrics if m in df.columns and
                     not df[m].isna().all() and not (df[m] == 0).all()]

    if not valid_metrics:
        print(f"  - Skipping {title} (no valid metrics)")
        return

    n_metrics = len(valid_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, metric in enumerate(valid_metrics):
        ax = axes[idx]
        color = colors[idx % len(colors)]

        # Plot metric with markers every 20 epochs
        checkpoint_mask = df['epoch'] % checkpoint_interval == 0

        # Main line
        ax.plot(df['epoch'], df[metric], '-', linewidth=2.5,
               alpha=0.8, color=color, label=get_metric_display_name(metric))

        # Markers at checkpoints
        ax.plot(df.loc[checkpoint_mask, 'epoch'],
               df.loc[checkpoint_mask, metric],
               'o', markersize=5, color=color, alpha=0.9)

        # Find best checkpoint
        lower_better = is_lower_better(metric)
        if lower_better:
            best_idx = checkpoint_df[metric].idxmin()
            metric_type = "Min"
        else:
            best_idx = checkpoint_df[metric].idxmax()
            metric_type = "Max"

        best_epoch = checkpoint_df.loc[best_idx, 'epoch']
        best_value = checkpoint_df.loc[best_idx, metric]

        # Mark best checkpoint with a star
        ax.plot(best_epoch, best_value, '*', markersize=20, color='red',
               markeredgecolor='darkred', markeredgewidth=2, zorder=5)

        # Add annotation
        y_range = df[metric].max() - df[metric].min()
        if y_range > 0:
            offset_y = -40 if best_value > df[metric].median() else 40
        else:
            offset_y = 40

        ax.annotate(f'{metric_type}: {best_value:.5f}\nEpoch {int(best_epoch)}',
                   xy=(best_epoch, best_value),
                   xytext=(0, offset_y), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.6', fc='#ffffcc',
                           edgecolor='darkred', linewidth=2, alpha=0.95),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkred'),
                   fontsize=11, fontweight='bold', ha='center', zorder=10)

        # Styling
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(get_metric_display_name(metric), fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(df['epoch'].min(), df['epoch'].max())

        # Add some padding to y-axis
        if y_range > 0:
            y_margin = y_range * 0.15
            ax.set_ylim(df[metric].min() - y_margin, df[metric].max() + y_margin)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ {title} saved to {filename}")


def plot_training_validation_comparison(df, checkpoint_df, output_dir):
    """Plot training vs validation loss comparison."""
    if 'train_loss' not in df.columns or 'val_loss' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot both losses
    ax.plot(df['epoch'], df['train_loss'], '-o', linewidth=2.5, markersize=4,
           alpha=0.8, color='#1f77b4', label='Training Loss', markevery=20)
    ax.plot(df['epoch'], df['val_loss'], '-s', linewidth=2.5, markersize=4,
           alpha=0.8, color='#ff7f0e', label='Validation Loss', markevery=20)

    # Find best validation checkpoint
    best_idx = checkpoint_df['val_loss'].idxmin()
    best_epoch = checkpoint_df.loc[best_idx, 'epoch']
    best_val_loss = checkpoint_df.loc[best_idx, 'val_loss']
    best_train_loss = df.loc[df['epoch'] == best_epoch, 'train_loss'].values[0]

    # Mark best checkpoint
    ax.plot(best_epoch, best_val_loss, '*', markersize=20, color='red',
           markeredgecolor='darkred', markeredgewidth=2, zorder=5)

    # Add annotation
    ax.annotate(f'Best Val Loss: {best_val_loss:.5f}\nTrain Loss: {best_train_loss:.5f}\nEpoch {int(best_epoch)}',
               xy=(best_epoch, best_val_loss),
               xytext=(30, -50), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.6', fc='#ffffcc',
                       edgecolor='darkred', linewidth=2, alpha=0.95),
               arrowprops=dict(arrowstyle='->', lw=2, color='darkred'),
               fontsize=12, fontweight='bold', zorder=10)

    # Styling
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=14)
    ax.set_title('Training vs Validation Loss', fontweight='bold', fontsize=18, pad=15)
    ax.legend(loc='best', fontsize=13, framealpha=0.95, edgecolor='black', linewidth=1.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(df['epoch'].min(), df['epoch'].max())

    plt.tight_layout()
    output_path = output_dir / 'training_vs_validation_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Training vs Validation comparison saved")


def plot_auxiliary_metrics(df, output_dir):
    """Plot learning rate and time metrics."""
    aux_metrics = []
    if 'learning_rate' in df.columns:
        aux_metrics.append('learning_rate')
    if 'epoch_time' in df.columns:
        aux_metrics.append('epoch_time')
    if 'validation_time' in df.columns:
        aux_metrics.append('validation_time')

    if not aux_metrics:
        return

    n_metrics = len(aux_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ['#9467bd', '#8c564b', '#e377c2']

    for idx, metric in enumerate(aux_metrics):
        ax = axes[idx]
        color = colors[idx % len(colors)]

        ax.plot(df['epoch'], df[metric], '-', linewidth=2.5,
               alpha=0.8, color=color)
        ax.fill_between(df['epoch'], df[metric], alpha=0.3, color=color)

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(get_metric_display_name(metric), fontweight='bold')
        ax.set_title(get_metric_display_name(metric), fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(df['epoch'].min(), df['epoch'].max())

    plt.tight_layout()
    output_path = output_dir / 'auxiliary_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Auxiliary metrics saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot training log metrics with publication-quality output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (plots saved to <csv_dir>/plots/)
  python plot_training_logs.py -i training_log.csv

  # Specify output directory
  python plot_training_logs.py -i training_log.csv -o ./my_plots

  # Custom checkpoint interval (default is 20)
  python plot_training_logs.py -i training_log.csv -o ./plots -c 10
        """
    )

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input CSV file with training logs')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for plots (default: <csv_dir>/plots/)')
    parser.add_argument('-c', '--checkpoint-interval', type=int, default=20,
                        help='Checkpoint interval in epochs (default: 20)')

    args = parser.parse_args()

    plot_training_logs(args.input, args.output, args.checkpoint_interval)
