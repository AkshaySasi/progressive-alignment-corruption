"""
Multi-seed experiment runner.

Runs the full corruption pipeline (data -> train -> evaluate) once per random
seed, writing each seed's outputs to a namespaced directory
(outputs/seed_<seed>/...). After all seeds finish, aggregates the key scalar
metrics across seeds into mean +/- std +/- 95% CI.

This addresses the single-seed limitation of the original study: the original
confidence intervals are bootstrap-over-prompts (sampling noise of a *single*
training run), whereas across-seed statistics capture training stochasticity
(data order, LoRA initialization, dropout), which is what reviewers expect for
claims about systematic degradation.

Usage:
    python -m src.run_multiseed --config config/config.yaml
    python -m src.run_multiseed --config config/config.yaml --seeds 42 123 456
    python -m src.run_multiseed --aggregate-only   # just re-aggregate

Each seed is independent and resumable: if a seed's outputs already exist they
are skipped (the underlying pipeline checks model_index.json and result files).
"""

import os
import sys
import copy
import json
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import (
    load_config,
    set_seed,
    step_build_datasets,
    step_train_models,
    step_evaluate_models,
)
from src.analysis.aggregate_seeds import aggregate_seeds


# Default seeds — same set documented in the Colab multi-seed plan.
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def run_single_seed(base_config: dict, seed: int, base_output_dir: str) -> str:
    """Run the full pipeline for one seed into a namespaced output dir."""
    config = copy.deepcopy(base_config)

    # Namespace every output under outputs/seed_<seed>/
    seed_output_dir = str(Path(base_output_dir) / f"seed_{seed}")
    config["experiment"]["output_dir"] = seed_output_dir
    config["experiment"]["seed"] = seed

    # Propagate the seed into training so the Trainer actually varies per seed.
    config.setdefault("training", {})["seed"] = seed

    print("\n" + "#" * 70)
    print(f"# SEED {seed}  ->  {seed_output_dir}")
    print("#" * 70)

    set_seed(seed)

    step_build_datasets(config)
    model_paths = step_train_models(config)
    step_evaluate_models(config, model_paths)

    return seed_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed runner for alignment corruption study"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Random seeds to run (default: 42 123 456 789 1024)",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip running; just aggregate existing seed_* outputs",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    base_output_dir = base_config["experiment"]["output_dir"]

    if not args.aggregate_only:
        for seed in args.seeds:
            run_single_seed(base_config, seed, base_output_dir)

    print("\n" + "=" * 70)
    print("AGGREGATING ACROSS SEEDS")
    print("=" * 70)
    agg = aggregate_seeds(
        base_output_dir=base_output_dir,
        seeds=args.seeds,
        corruption_types=base_config["dataset"]["corruption_types"],
    )
    print(f"\nAggregated {agg['n_seeds_found']} seed(s).")
    print(f"Tables written to: {agg['output_dir']}")


if __name__ == "__main__":
    main()
