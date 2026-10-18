#!/usr/bin/env python3
"""
Run a specific FairDiverse experiment based on task array ID.
This script is called by the SLURM job array.
"""
import json
import yaml
import os
import sys
import argparse
import tempfile
from pathlib import Path


def _log_name_component(value):
    """Convert a config value into a filesystem- and W&B-friendly name part."""
    component = str(value).strip().split("/")[-1]
    return "-".join(component.replace("_", "-").split())


def resolve_auto_log_name(config):
    """Build a GRID run name including dataset and representation provenance."""
    log_name = config.get("log_name")
    dataset = config.get("dataset")
    if log_name != "auto":
        prefix = "grid-independent-"
        dataset_component = _log_name_component(dataset) if dataset else ""
        expected_prefix = f"{prefix}{dataset_component}-"
        if (
            isinstance(log_name, str)
            and log_name.startswith(prefix)
            and dataset_component
            and not log_name.startswith(expected_prefix)
        ):
            config["log_name"] = expected_prefix + log_name[len(prefix):]
        return

    values = (
        "grid-independent",
        dataset,
        config.get("grid_tokenizer", config.get("tokenizers")),
        config.get("embedding_model"),
        config.get("metadata_text_mode"),
        config.get("rank_model"),
    )
    config["log_name"] = "-".join(
        _log_name_component(value) for value in values if value not in (None, "")
    )


def load_experiments(config_file='experiments_config.json'):
    """Load all experiments from JSON config file."""
    with open(config_file, 'r') as f:
        experiments = json.load(f)
    return experiments


def run_experiment(experiment, topk_override=None, use_8bit_override=False, sid_path_override=None, seed_override=None, level_loss_weights_override=None, slurm_job_id=None, slurm_array_task_id=None):
    """Run a single experiment."""
    task = experiment['task']
    stage = experiment['stage']
    dataset = experiment['dataset']
    config = experiment['config']
    
    # Add SLURM tracking info to config (will be logged to WandB)
    if slurm_job_id:
        config['slurm_job_id'] = slurm_job_id
    if slurm_array_task_id:
        config['slurm_array_task_id'] = slurm_array_task_id
    
    # Merge top-level fields into config for trainer compatibility
    # Trainers expect task path relative to fairdiverse/ directory
    config['task'] = f'fairdiverse/{task}'
    config['stage'] = stage
    config['dataset'] = dataset
    resolve_auto_log_name(config)
    
    # Validate required attributes based on task and stage
    required_attrs = []
    if task == "recommendation":
        if stage == "in-processing":
            required_attrs = ['model', 'data_type', 'fair-rank', 'log_name',
                            'device', 'epoch', 'batch_size', 'learning_rate',
                            'topk']
            if config.get('fair-rank', False):
                required_attrs += ['rank_model', 'watch_metric', 'eval_step', 'fairness_type']
        elif stage == "post-processing":
            required_attrs = ['model', 'rank_model', 'log_name', 'topk', 'fairness_type']
    elif task == "search":
        if stage == "post-processing":
            required_attrs = ['model', 'rank_model', 'log_name', 'topk']
        elif stage == "pre-processing":
            required_attrs = ['model', 'log_name', 'epoch', 'batch_size', 'learning_rate']
    
    # Check for missing required attributes
    if required_attrs:
        missing = [attr for attr in required_attrs if attr not in config]
        if missing:
            print(f"⚠ Warning: Missing required attributes for {task}/{stage}: {missing}")
    
    # Log all config attributes
    print("\nConfiguration attributes read:")
    print("-" * 80)
    for key in sorted(config.keys()):
        value = config[key]
        if any(word in key.lower() for word in ("api_key", "access_token", "auth_token", "hf_token", "password", "secret")):
            value = "[redacted]"
        # Truncate long values for readability
        if isinstance(value, str) and len(value) > 60:
            display_value = value[:60] + "..."
        else:
            display_value = value
        print(f"  {key}: {display_value}")
    print("-" * 80 + "\n")
    
    # Override topk if provided
    if topk_override is not None:
        config['topk'] = topk_override
        print(f"✓ Overriding topk to: {topk_override}")
    
    # Override sid_path if provided
    if sid_path_override is not None:
        # Support short names ("sids_rvq_t5xl.pt"), relative ("GRID/outputs/..."), or absolute ("/abs/path/...")
        if sid_path_override.startswith("/") or sid_path_override.startswith("GRID/outputs/"):
            # Already an absolute or full relative path — use as-is
            pass
        else:
            # Just a filename — prepend the dataset output dir
            dataset_name = dataset if dataset != "amazon-beauty" else "amazon-beauty"
            sid_path_override = f"GRID/outputs/{dataset_name}/{sid_path_override}"
        config['sid_path'] = sid_path_override
        print(f"✓ Overriding sid_path to: {sid_path_override}")
    
    # Override level_loss_weights if provided via command line
    if level_loss_weights_override is not None:
        config['level_loss_weights'] = level_loss_weights_override
        print(f"✓ Overriding level_loss_weights to: {level_loss_weights_override}")

    # Override seed if provided via command line
    if seed_override is not None:
        config['seed'] = seed_override
        # Append seed suffix to log_name so each trial saves to a separate directory
        base_log_name = config.get('log_name', 'unnamed')
        config['log_name'] = f"{base_log_name}-s{seed_override}"
        print(f"✓ Overriding seed to: {seed_override}  (log_name → {config['log_name']})")

    # Apply 8-bit quantization flag if specified
    if use_8bit_override:
        config['use_8bit'] = True
        print(f"✓ Enabling 8-bit quantization for LLM model")
    
    # If using LLM, ensure required LLM config parameters are set
    if config.get('use_llm', False):
        # Set default LLM configuration if not already present
        if 'llm_type' not in config:
            config['llm_type'] = 'vllm'  # Default to vllm for efficiency
            print(f"⚠ Setting llm_type to 'vllm' (default)")
        
        if 'llm_name' not in config:
            # Use model name as llm_name if not set
            config['llm_name'] = config.get('model', 'Llama3')
            print(f"⚠ Setting llm_name to '{config['llm_name']}'")
        
        # Set fair_prompt for FairPrompts model
        if 'fair_prompt' not in config:
            config['fair_prompt'] = "You are a item-fair recommender. Please try to ensure that each category of items receives fair recommendations."
            print(f"⚠ Setting fair_prompt (default)")
        
        # Set other LLM defaults if not present
        if 'use_8bit' not in config:
            config['use_8bit'] = use_8bit_override  # Use command-line flag
        if 'device_map' not in config:
            config['device_map'] = 'auto'
        if 'max_tokens' not in config:
            config['max_tokens'] = 256
        if 'temperature' not in config:
            config['temperature'] = 0.8
        if 'use_batch' not in config:
            config['use_batch'] = False
    
    print("="*80)
    print(f"Running experiment: {config.get('log_name', 'unnamed')}")
    print(f"Task: {task}")
    print(f"Stage: {stage}")
    print(f"Dataset: {dataset}")
    print("="*80)
    
    # Create temporary YAML config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir='.') as tmp_file:
        yaml.dump(config, tmp_file, default_flow_style=False)
        tmp_config_path = tmp_file.name
    
    try:
        # Import and run the appropriate trainer/reranker
        if task == "recommendation":
            if stage == 'preprocessing':
                # Standalone preprocessing: build processed_dataset/ only, no training.
                # Run this as a separate job before any in-processing or post-processing
                # jobs so they can safely run in parallel without a race condition.
                processed_dir = os.path.join("recommendation", "processed_dataset", dataset)
                if os.path.exists(processed_dir) and not config.get('reprocess', False):
                    print(f"✓ Processed dataset already exists at '{processed_dir}'. Nothing to do.")
                else:
                    from fairdiverse.recommendation.process_dataset import Process
                    result = Process(dataset, config)
                    print(result)
            elif stage == 'in-processing':
                # Native GRID runs can build the FairDiverse tables from the
                # embedded temporal-LOO data inside RecTrainer. Other models
                # still require an explicit preprocessing job.
                processed_dir = os.path.join("recommendation", "processed_dataset", dataset)
                builds_from_embedded_grid = (
                    config.get('model') == 'GRID'
                    and config.get('grid_data_source') == 'embedded_tfrecords'
                )
                if not os.path.exists(processed_dir) and not builds_from_embedded_grid:
                    raise RuntimeError(
                        f"Processed dataset not found at '{processed_dir}'.\n"
                        f"Run the 'preprocessing' stage experiment for dataset '{dataset}' first, "
                        f"then resubmit this job."
                    )
                if config.get('use_llm', False):
                    from fairdiverse.recommendation.llm_rec import LLMRecommender
                    recommender = LLMRecommender(config)
                    recommender.recommend()
                else:
                    seed = config.get('seed', None)
                    if seed is not None:
                        import random
                        import numpy as np
                        import torch
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        print(f"✓ Random seed set to {seed}")
                    from fairdiverse.recommendation.trainer import RecTrainer
                    trainer = RecTrainer(config)
                    trainer.train()
            elif stage == 'post-processing':
                from fairdiverse.recommendation.reranker import RecReRanker
                reranker = RecReRanker(config)
                reranker.rerank()
            else:
                raise NotImplementedError(f"Stage {stage} not supported")
        
        elif task == "search":
            if stage == "post-processing":
                if dataset != 'clueweb09':
                    raise ValueError("For post-processing methods in search, we only support the clueweb09 dataset")
                from fairdiverse.search.trainer import SRDTrainer
                trainer = SRDTrainer(config)
                trainer.train()
            elif stage == "pre-processing":
                from fairdiverse.search.trainer_preprocessing_ranker import RankerTrainer
                trainer = RankerTrainer(config)
                trainer.train()
            else:
                raise NotImplementedError(f"Stage {stage} not supported")
        else:
            raise NotImplementedError(f"Task {task} not supported")
        
        print("\n" + "="*80)
        print(f"✓ Experiment completed successfully: {config.get('log_name', 'unnamed')}")
        print("="*80)
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ Experiment failed: {config.get('log_name', 'unnamed')}")
        print(f"Error: {str(e)}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary config file
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)


def select_experiment(experiments, exp_id):
    """Select an explicit experiment ID, falling back to index for legacy entries."""
    matches = [entry for index, entry in enumerate(experiments)
               if entry.get('exp_id', index) == exp_id]
    if len(matches) != 1:
        raise ValueError(f"Experiment ID {exp_id} must match exactly one entry; found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description='Run FairDiverse experiments')
    parser.add_argument('--exp_id', type=int, required=True, 
                        help='Explicit exp_id, or 0-based index for entries without exp_id')
    parser.add_argument('--topk', type=str, default=None,
                        help='Override topk values as JSON list, e.g., "[10, 20]"')
    parser.add_argument('--config_file', type=str, default='experiments_config.json',
                        help='Path to experiments config file')
    parser.add_argument('--use_8bit', action='store_true',
                        help='Enable 8-bit quantization for LLM models to reduce memory usage')
    parser.add_argument('--sid_path_override', type=str, default=None,
                        help='Override sid_path in config, e.g., "sids_rvq_t5xl.pt" or "GRID/outputs/steam/sids_rvq_t5xl.pt"')
    parser.add_argument('--slurm_job_id', type=str, default=None,
                        help='SLURM job ID for tracking')
    parser.add_argument('--slurm_array_task_id', type=str, default=None,
                        help='SLURM array task ID for tracking')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed in config. Appends -s{seed} to log_name so each trial saves to a separate directory.')
    parser.add_argument('--level_loss_weights', type=str, default=None,
                        help='Override level_loss_weights as JSON list, e.g., "[4,2,1]" to weight SID hierarchy levels.')
    args = parser.parse_args()
    
    # Load experiments
    experiments = load_experiments(args.config_file)
    
    try:
        experiment = select_experiment(experiments, args.exp_id)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    
    # Parse topk override if provided
    topk_override = None
    if args.topk:
        try:
            topk_override = json.loads(args.topk)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse topk '{args.topk}', using config default")
    
    # Apply 8-bit quantization flag if specified
    use_8bit_override = args.use_8bit
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {args.exp_id + 1}/{len(experiments)}")
    print(f"{'='*80}\n")
    
    # Run the experiment
    level_loss_weights_override = None
    if args.level_loss_weights:
        try:
            level_loss_weights_override = json.loads(args.level_loss_weights)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse level_loss_weights '{args.level_loss_weights}', using config default")

    success = run_experiment(experiment, topk_override=topk_override, use_8bit_override=use_8bit_override,
                           sid_path_override=args.sid_path_override,
                           seed_override=args.seed,
                           level_loss_weights_override=level_loss_weights_override,
                           slurm_job_id=args.slurm_job_id, slurm_array_task_id=args.slurm_array_task_id)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
