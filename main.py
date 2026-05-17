import argparse
from pathlib import Path

from dcdm.config import config

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the D-CDM denoising network.")
    parser.add_argument("--graph-dir", type=Path, default=PROJECT_ROOT / "graph")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--train-eval-every", type=int, default=1)
    parser.add_argument("--comparison-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--dataset-check-samples", type=int, default=50)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening matplotlib windows.")
    return parser.parse_args()


def resolve_resume_path(args, checkpoints_dir):
    if args.resume is not None:
        return args.resume.expanduser().resolve()
    if args.resume_latest:
        from dcdm.checkpointing import find_latest_checkpoint

        return find_latest_checkpoint(checkpoints_dir)
    return None


def check_required_package(import_name, install_hint):
    try:
        __import__(import_name)
        return None
    except ModuleNotFoundError:
        return f"{import_name} is not installed. {install_hint}"


def validate_training_prerequisites(graph_dir):
    from dcdm.datasets import require_dataset_files, resolve_dataset_paths

    errors = []
    package_checks = [
        ("torch", "Install PyTorch for your CUDA/CPU environment."),
        (
            "torch_geometric",
            "Install a torch-geometric build that matches your PyTorch and CUDA versions.",
        ),
    ]
    for import_name, install_hint in package_checks:
        error = check_required_package(import_name, install_hint)
        if error:
            errors.append(error)

    try:
        require_dataset_files(resolve_dataset_paths(graph_dir))
    except FileNotFoundError as exc:
        errors.append(str(exc))

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(f"Training prerequisites are not ready:\n{details}")


def main():
    args = parse_args()

    from tqdm import tqdm

    from dcdm.checkpointing import (
        load_training_state,
        prune_checkpoints,
        save_checkpoint,
        save_final_model,
        save_if_best,
    )
    from dcdm.datasets import load_dataset_bundle, print_dataset_report
    from dcdm.metrics import (
        Stopwatch,
        TrainingHistory,
        current_learning_rate,
        format_epoch_record,
        make_epoch_record,
    )
    from dcdm.runtime import (
        ProjectPaths,
        ensure_directories,
        print_run_header,
        set_random_seed,
        summarize_model,
    )

    config.batch_size = args.batch_size
    paths = ProjectPaths.build(PROJECT_ROOT, args.graph_dir, args.output_dir)
    ensure_directories(paths)
    set_random_seed(args.seed, deterministic=args.deterministic)
    print_run_header(config, paths, args.seed)
    validate_training_prerequisites(paths.graph_dir)

    from dcdm.evaluation import evaluate_model
    from dcdm.trainer import DCDM_Trainer
    from dcdm.visualization import visualize_g0_comparison

    bundle = load_dataset_bundle(paths.graph_dir, max_check_samples=args.dataset_check_samples)
    print_dataset_report(bundle)

    trainer = DCDM_Trainer(config, device=args.device)
    model_stats = summarize_model(trainer.denoising_net)
    print("\n=== Model Report ===")
    print(f"Total parameters: {model_stats['parameters_total']:,}")
    print(f"Trainable parameters: {model_stats['parameters_trainable']:,}")

    history = TrainingHistory()
    best_reward = None
    start_epoch = 1
    resume_path = resolve_resume_path(args, paths.checkpoints_dir)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, best_reward, history = load_training_state(
            trainer,
            resume_path,
            map_location=str(trainer.device),
        )
        print(f"\nResumed from checkpoint: {resume_path}")
        print(f"Next epoch: {start_epoch}")

    if start_epoch > args.epochs:
        print(f"\nNo training needed: start epoch {start_epoch} is beyond --epochs {args.epochs}.")
    else:
        progress = tqdm(range(start_epoch, args.epochs + 1), desc="Training D-CDM")
        for epoch in progress:
            timer = Stopwatch()
            loss = trainer.train_step(bundle.train)
            rewards = []

            should_sample_rewards = (
                args.train_eval_every > 0
                and (epoch % args.train_eval_every == 0 or epoch == args.epochs)
            )
            if should_sample_rewards:
                _, rewards = trainer.sample_trajectory_from_G0(bundle.train, batch_size=args.eval_batch_size)

            record = make_epoch_record(
                epoch=epoch,
                loss=loss,
                rewards=rewards,
                elapsed_sec=timer.elapsed(),
                learning_rate=current_learning_rate(trainer.optimizer),
            )
            history.append(record)
            print(format_epoch_record(record))
            history.save_csv(paths.history_path)

            if record.reward_valid_count > 0:
                best_reward, best_path = save_if_best(
                    trainer=trainer,
                    epoch=epoch,
                    history=history,
                    checkpoints_dir=paths.checkpoints_dir,
                    best_reward=best_reward,
                    candidate_reward=record.reward_valid_mean,
                    config=config,
                )
                if best_path is not None:
                    print(f"New best checkpoint saved: {best_path}")

            if args.save_every > 0 and epoch % args.save_every == 0:
                checkpoint_path = save_checkpoint(
                    trainer=trainer,
                    epoch=epoch,
                    history=history,
                    checkpoints_dir=paths.checkpoints_dir,
                    best_reward=best_reward,
                    config=config,
                )
                print(f"Checkpoint saved: {checkpoint_path}")
                removed = prune_checkpoints(paths.checkpoints_dir, keep=args.keep_checkpoints)
                for old_path in removed:
                    print(f"Removed old checkpoint: {old_path}")

    print("\n=== Training Summary ===")
    print(history.short_report())
    model_path = save_final_model(trainer, paths.model_path)
    print(f"Training completed! Model saved as '{model_path}'")

    if not args.skip_eval:
        evaluate_model(
            trainer,
            bundle.test,
            batch_size=args.eval_batch_size,
            num_batches=args.eval_batches,
            save_path=paths.figure_path("fig9_reward_curve.png"),
            show=not args.no_show,
        )

    if not args.skip_visualization:
        print("\n=== Generating G0 Comparison Visualization ===")
        visualize_g0_comparison(
            trainer,
            bundle.train,
            num_samples=args.comparison_samples,
            save_path=paths.figure_path("g0_comparison.png"),
            show=not args.no_show,
        )
        print("G0 comparison visualization completed!")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
