import numpy as np
#from
import os

import pandas as pd
import yaml
import random

from .process_dataset import Process
from .base_model import MF, GRU4Rec, SASRec, SASRecSPRINT, BPR, BPR_Seq, GRIDBaseModel
from .base_model.grid_collate import collate_fn_grid_causal_augmentation
from .grid_data_bridge import ensure_grid_fairdiverse_data
from .rank_model import IPS, SDRO, Minmax_SGD, APR, FOCF, FairDual, Reg, FairNeg, DPR

import time
import torch.optim as optim

from .sampler import (
    PointWiseDataset,
    PairWiseDataset,
    RankingTestDataset,
    SequentialDataset,
    UserGroupedBatchSampler,
)
from .evaluator import CTR_Evaluator, Ranking_Evaluator

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm,trange
from datetime import datetime
import torch
import json
from scipy.sparse import save_npz, load_npz
import torch.nn as nn
import ast
import fcntl

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")



class RecTrainer(object):
    def __init__(self, train_config):
        """Initialize In-processing and base models.

        :param train_config: Your custom config files.
        """

        self.dataset = train_config['dataset']
        #self.stage = stage
        self.train_config = train_config
        # Get the directory where trainer.py is located (fairdiverse/recommendation/)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def _seed_everything(seed: int, deterministic: bool = False) -> None:
        """Seed Python, NumPy, Torch, CUDA, and cuDNN flags for reproducible runs."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic

        print(f"[SEED] seed={seed}, deterministic={deterministic}")

    @staticmethod
    def _make_worker_init_fn(base_seed: int):
        """Return a worker init fn so DataLoader subprocesses get deterministic RNG state."""
        def _seed_worker(worker_id: int) -> None:
            worker_seed = base_seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        return _seed_worker


    def load_configs(self, dir):
        """
            Loads and merges configuration files for the model, dataset, and evaluation.

            This function loads multiple YAML configuration files, including the process configuration,
            dataset-specific settings, model configurations, and evaluation parameters. All configurations
            are merged, with the highest priority given to the class's own `config` attribute.

            :param dir: The directory where the main process configuration file is located.
            :return: A dictionary containing the merged configuration from all files.
        """

        # current_dir = os.getcwd()
        # print(f"Current directory: {current_dir}")
        
        print("start to load config...")
        with open(os.path.join(dir, "process_config.yaml"), 'r') as f:
            config = yaml.safe_load(f)

        # A processed dataset may have been produced by the independent GRID
        # bridge.  Its process_config also records the GRID run settings for
        # reproducibility, but those settings are not model-agnostic dataset
        # properties.  Do not let a later SASRec/GRU/etc. run silently inherit
        # GRID's optimizer schedule, architecture, checkpoint, or evaluator
        # mode.  Explicit values in self.train_config are merged below and can
        # still opt into any generic setting intentionally.
        if self.train_config.get('model') != 'GRID':
            grid_model_only_keys = {
                'accumulate_grad_batches',
                'beam_diagnostics',
                'causal_group_by_user',
                'causal_pre_batch_size',
                'codebook_size',
                'd_kv',
                'early_stop_metrics',
                'early_stop_min_delta',
                'early_stop_patience',
                'early_stop_weights',
                'embedding_model',
                'grid_checkpoint_path',
                'grid_eval_only',
                'grid_eval_generation_batch_size',
                'grid_fair_finetune',
                'grid_fair_finetune_steps',
                'grid_native_diagnostics',
                'grid_native_metrics',
                'grid_tokenizer',
                'level_loss_weights',
                'max_steps',
                'mlp_layers',
                'num_hierarchies',
                'num_user_bins',
                'ranking_loss_weight',
                'sid_path',
                'tokenizers',
                'top_k_for_generation',
                'use_beam_search_eval',
                'use_causal_augmentation',
                'use_prefix_validation',
                'val_check_interval_batches',
                'weight_decay',
                'weight_tying',
            }
            for key in grid_model_only_keys:
                config.pop(key, None)


        # print(train_data_df.head())

        print("start to load model...")
        with open(os.path.join(self.base_dir, "properties", "models.yaml"), 'r') as f:
            model_config = yaml.safe_load(f)

        # Model name mapping: some files are lowercase, some are mixed case
        # Map: config name -> actual filename (without .yaml extension)
        model_filename_mapping = {
            'mf': 'mf',
            'MF': 'mf',
            'BPR': 'BPR',
            'BPR_Seq': 'BPR_Seq',
            'gru4rec': 'gru4rec',
            'GRU4Rec': 'gru4rec',
            'SASRec': 'SASRec',
            'SASRecSPRINT': 'SASRecSPRINT',
            'GRID': 'GRID',
        }
        model_filename = model_filename_mapping.get(self.train_config['model'], self.train_config['model'])
        with open(os.path.join(self.base_dir, "properties", "models", model_filename + ".yaml"),
                  'r') as f:
            model_config.update(yaml.safe_load(f))

        if self.train_config['fair-rank'] == True:
            with open(os.path.join(self.base_dir, "properties", "models", self.train_config['rank_model'] + ".yaml"),
                      'r') as f:
                model_config.update(yaml.safe_load(f))

        config.update(model_config)

        with open(os.path.join(self.base_dir, "properties", "evaluation.yaml"), 'r') as f:
            config.update(yaml.safe_load(f))

        config.update(self.train_config)  ###train_config has highest rights

        # A base run must not retain an inherited fair-ranker default (for
        # example APR) in its effective configuration.  Fair-ranking behavior
        # is controlled exclusively by fair-rank.
        if not config.get('fair-rank', False):
            config.pop('rank_model', None)

        print("your loading config is:")
        print(config)

        return config

    def Set_Dataset(self, data_type, config, train_data_df, val_data_df, test_data_df):
        """
            Initializes and returns the training, validation, and test datasets based on the specified data type and evaluation type.

            This function creates appropriate dataset objects for training, validation, and testing based on the provided data type
            (point, pair, or sequential) and the evaluation type (CTR or ranking). It supports different dataset types for training
            and evaluation, and raises an error if an unsupported type is provided.

            :param data_type: The type of dataset to be used for training. Must be one of ['point', 'pair', 'sequential'].
            :param config: A configuration dictionary that contains parameters for dataset creation and evaluation type.
            :param train_data_df: The DataFrame containing the training data.
            :param val_data_df: The DataFrame containing the validation data.
            :param test_data_df: The DataFrame containing the test data.
            :return: A tuple containing the training, validation, and test datasets.
        """
        if data_type == 'point':
            train = PointWiseDataset(train_data_df, config)
        elif data_type == 'pair':
            train = PairWiseDataset(train_data_df, config)
        elif data_type == 'sequential':
            train = SequentialDataset(train_data_df, config)
        else:
            raise NotImplementedError("train_type only supports in [point, pair, sequential]")

        if config['eval_type'] == 'CTR':
            valid = PointWiseDataset(val_data_df, config)
            test = PointWiseDataset(test_data_df, config)
        elif config['eval_type'] == 'ranking':
            valid = RankingTestDataset(val_data_df, config)
            test = RankingTestDataset(test_data_df, config)
        else:
            raise NotImplementedError("We only support the eval type as [CTR, ranking]")

        return train, valid, test

    def check_model_stage(self,config, Model):
        """
            Checks if the provided data type in the configuration aligns with the supported model type.

            This function verifies that the data type specified in the `config` dictionary is compatible with the model's
            supported types. If the data type is not supported by the model, a `ValueError` is raised with an informative message.

            :param config: A configuration dictionary that includes the data type used for testing.
            :param Model: The model class or object which has a `type` attribute specifying the supported data types.
        """
        if config['data_type'] not in Model.type:
            raise ValueError(f"The tested data type does not align with the model type: input is {config['data_type']}, "
                             f"the model only support: {Model.type}")

    def train(self):
        """
            Training in-processing and base model main workflow.
        """
        # current_dir = os.getcwd()
        dir = os.path.join("recommendation", "processed_dataset", self.dataset)

        # Matched baselines must not silently reuse a different metadata split.
        expected_source = self.train_config.get('expected_grid_source_dir')
        if expected_source:
            if self.train_config.get('reprocess', False):
                raise ValueError('Matched GRID baselines require reprocess=false')
            manifest_path = os.path.join(dir, 'grid_data_manifest.json')
            with open(manifest_path) as manifest_file:
                manifest = json.load(manifest_file)
            if (
                os.path.normpath(manifest.get('source_dir', ''))
                != os.path.normpath(expected_source)
                or manifest.get('split_strategy') != 'temporal_loo'
            ):
                raise ValueError(
                    f'{self.dataset}: processed data does not match '
                    f'temporal_loo source {expected_source}'
                )

        initial_seed = int(self.train_config.get('seed', 42))
        initial_deterministic = bool(self.train_config.get('deterministic', False))
        self._seed_everything(initial_seed, deterministic=initial_deterministic)

        # Independent GRID runs share one FairDiverse item mapping. Slurm
        # arrays may start many fairness methods simultaneously, so serialize
        # the one-time preprocessing pass instead of letting every task rewrite
        # recommendation/processed_dataset at once.
        if (
            self.train_config.get('model') == 'GRID'
            and self.train_config.get('grid_data_source') == 'embedded_tfrecords'
        ):
            processed_root = os.path.join('recommendation', 'processed_dataset')
            os.makedirs(processed_root, exist_ok=True)
            lock_path = os.path.join(processed_root, f'.{self.dataset}.grid-preprocess.lock')
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                state = ensure_grid_fairdiverse_data(self.train_config)
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        else:
            state = Process(self.dataset, self.train_config)
        config = self.load_configs(dir)
        print(state)
        #exit(0)

        run_seed = int(config.get('seed', initial_seed))
        run_deterministic = bool(config.get('deterministic', initial_deterministic))
        self._seed_everything(run_seed, deterministic=run_deterministic)


        print("start to load dataset......")
        # Load dataset statistics from preprocessed data
        stats_file = os.path.join("recommendation", "processed_dataset", config['dataset'], "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                dataset_stats = json.load(f)
            config['item_num'] = dataset_stats['item_num']
            config['user_num'] = dataset_stats['user_num']
            config['group_num'] = dataset_stats['group_num']
            print(f"Loaded dataset statistics: {dataset_stats}")
        else:
            raise ValueError(f"Dataset statistics file not found: {stats_file}. Please run preprocessing first.")

        if config.get('model') == 'GRID' and config.get('grid_item_mapping_path'):
            grid_mapping_path = os.path.realpath(config['grid_item_mapping_path'])
            embedded_grid_root = os.path.realpath('GRID') + os.sep
            if not grid_mapping_path.startswith(embedded_grid_root):
                raise ValueError(
                    "grid_item_mapping_path must point inside this repository's GRID directory"
                )
            fair_mapping_path = os.path.join(dir, 'iid2item.json')
            if not os.path.isfile(fair_mapping_path):
                raise FileNotFoundError(
                    f"Missing {fair_mapping_path}; rerun FairDiverse preprocessing "
                    "to record the item mapping."
                )
            with open(grid_mapping_path, 'r') as f:
                grid_item_to_iid = json.load(f)
            with open(fair_mapping_path, 'r') as f:
                fair_iid_to_item = json.load(f)
            mismatches = [
                (iid, original_item, grid_item_to_iid.get(str(original_item)))
                for iid, original_item in fair_iid_to_item.items()
                if grid_item_to_iid.get(str(original_item)) != int(iid)
            ]
            if mismatches or len(grid_item_to_iid) != len(fair_iid_to_item):
                raise ValueError(
                    "FairDiverse and GRID item mappings differ. First mismatches: "
                    f"{mismatches[:5]}"
                )
            print(f"[GRID] Verified identical item mapping for {len(fair_iid_to_item)} items")

        print("start to load dataset......")

        self.device = config['device']
        
        self.device = config['device']

        if config['model'] == 'mf':
            self.Model = MF(config).to(self.device)
        elif config['model'] == 'BPR':
            self.Model = BPR(config).to(self.device)
        elif config['model'] == 'BPR_Seq':
            self.Model = BPR_Seq(config).to(self.device)
        elif config['model'] in ('gru4rec', 'GRU4Rec'):
            self.Model = GRU4Rec(config).to(self.device)
        elif config['model'] == 'SASRec':
            self.Model = SASRec(config).to(self.device)
        elif config['model'] == 'SASRecSPRINT':
            self.Model = SASRecSPRINT(config).to(self.device)

        elif config['model'] == 'GRID':
            # GRIDBaseModel uses SID embeddings produced by the offline GRID pipeline.
            # Requires 'sid_path', 'num_hierarchies', and 'codebook_size' in config.
            self.Model = GRIDBaseModel(config).to(self.device)

        else:
            raise NotImplementedError(f"Not supported model type: {config['model']}")

        if config.get('grid_fair_finetune', False):
            if config.get('model') != 'GRID' or not config.get('fair-rank', False):
                raise ValueError(
                    "grid_fair_finetune requires model=GRID and fair-rank=true"
                )
            if config.get('grid_eval_only', False):
                raise ValueError("grid_fair_finetune cannot use grid_eval_only")
            if not getattr(self.Model, 'loaded_checkpoint_path', None):
                raise ValueError(
                    "grid_fair_finetune requires a successfully loaded native checkpoint"
                )
            requested_steps = int(config.get('grid_fair_finetune_steps', 0) or 0)
            if requested_steps <= 0:
                raise ValueError("grid_fair_finetune_steps must be positive")
            if int(config.get('max_steps', 0) or 0) != requested_steps:
                raise ValueError(
                    "For GRID fairness fine-tuning, max_steps must equal "
                    "grid_fair_finetune_steps"
                )
            print(
                "[GRID] Checkpoint-initialized fairness fine-tuning: "
                f"ranker={config['rank_model']}, optimizer_steps={requested_steps}, "
                f"checkpoint={self.Model.loaded_checkpoint_path}"
            )

        self.check_model_stage(config, self.Model)

        self.group_weight = np.ones(config['group_num'])





        train_data_df = pd.read_csv(os.path.join(dir, self.dataset + ".train"), sep='\t')
        val_data_df = pd.read_csv(os.path.join(dir, self.dataset + ".valid." + config['eval_type']), sep='\t')
        test_data_df = pd.read_csv(os.path.join(dir, self.dataset + ".test." + config['eval_type']), sep='\t')

        print(test_data_df.head())
        train_data_df["history_behaviors"] = train_data_df["history_behaviors"].apply(lambda x: np.array(ast.literal_eval(x)))
        val_data_df["history_behaviors"] = val_data_df["history_behaviors"].apply(lambda x: np.array(ast.literal_eval(x)))
        test_data_df["history_behaviors"] = test_data_df["history_behaviors"].apply(lambda x: np.array(ast.literal_eval(x)))

        optimizer = optim.Adam(
            self.Model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0),
        )
        _accum = config.get('accumulate_grad_batches', 1)
        data_type = config['data_type']
        worker_init_fn = self._make_worker_init_fn(run_seed)



        train, valid, test = self.Set_Dataset(data_type, config, train_data_df, val_data_df, test_data_df)

        if config['fair-rank'] == True:
            if config['data_type'] == 'point':
                raise ValueError(
                    "fair ranking model only supports the pair and sequential data_type, not the point type")

            if config['rank_model'] == "IPS":
                self.Fair_Ranker = IPS(config, self.group_weight)

            elif config['rank_model'] == 'SDRO':
                self.Fair_Ranker = SDRO(config, self.group_weight)

            elif config['rank_model'] == 'Minmax_SGD':
                self.Fair_Ranker = Minmax_SGD(config, self.group_weight)

            elif config['rank_model'] == 'APR':
                self.Fair_Ranker = APR(config, self.group_weight)

            elif config['rank_model'] == 'FOCF':
                self.Fair_Ranker = FOCF(config, self.group_weight)

            elif config['rank_model'] == 'FairDual':
                self.Fair_Ranker = FairDual(config, self.group_weight)

            elif config['rank_model'] == 'Reg':
                self.Fair_Ranker = Reg(config, self.group_weight)

            elif config['rank_model'] == 'FairNeg':
                self.Fair_Ranker = FairNeg(config, train.user2pos)

            elif config['rank_model'] == 'DPR':
                self.Fair_Ranker = DPR(config, self.group_weight)

            else:
                NotImplementedError(f"Not supported fair rank model type:{config['rank_model']}")

            self.check_model_stage(config, self.Fair_Ranker)
            if self.Fair_Ranker.fair_type == "sample" and config['data_type'] != "pair":
                raise ValueError(
                    f"The choosed fair ranker [{config['rank_model']}] only support the base model type as pair")


        # For GRID with causal augmentation, the DataLoader uses a smaller
        # pre-augmentation batch size; the collate function expands it up to
        # config['batch_size'] samples by generating all causal sub-sequences.
        # This mirrors original GRID's collate_with_sid_causal_duplicate.
        if config.get('model') == 'GRID' and config.get('use_causal_augmentation', False):
            pre_batch_size = config.get('causal_pre_batch_size', 32)
            max_aug_batch  = config['batch_size']
            _collate_fn = lambda batch: collate_fn_grid_causal_augmentation(
                batch,
                max_batch_size=max_aug_batch,
                padding_value=config.get('padding_item_id', 0),
            )
            num_workers = config.get('dataloader_num_workers', 4)
            grouped_by_user = config.get('causal_group_by_user', False)
            batch_sampler = None
            if grouped_by_user:
                batch_sampler = UserGroupedBatchSampler(
                    train.user_ids,
                    users_per_batch=pre_batch_size,
                    seed=run_seed,
                )
                train_loader = DataLoader(
                    train,
                    batch_sampler=batch_sampler,
                    collate_fn=_collate_fn,
                    num_workers=num_workers,
                    persistent_workers=(num_workers > 0),
                    worker_init_fn=worker_init_fn,
                )
            else:
                train_loader = DataLoader(
                    train,
                    batch_size=pre_batch_size,
                    shuffle=True,
                    collate_fn=_collate_fn,
                    num_workers=num_workers,
                    persistent_workers=(num_workers > 0),
                    worker_init_fn=worker_init_fn,
                )
            print(
                f"[GRID] Causal augmentation enabled: "
                f"pre_batch={pre_batch_size} {'users' if grouped_by_user else 'rows'}, "
                f"max_post_batch={max_aug_batch}, grouped_by_user={grouped_by_user}"
            )
        else:
            train_loader = DataLoader(
                train,
                batch_size=config['batch_size'],
                shuffle=True,
                worker_init_fn=worker_init_fn,
            )
        valid_loader = DataLoader(
            valid,
            batch_size=config['eval_batch_size'],
            shuffle=False,
            worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            test,
            batch_size=config['eval_batch_size'],
            shuffle=False,
            worker_init_fn=worker_init_fn,
        )

        if config['eval_type'] == 'CTR':
            evaluator = CTR_Evaluator(config)
        elif config['eval_type'] == 'ranking':
            evaluator = Ranking_Evaluator(config)
        else:
            raise NotImplementedError("we only support eval type in [CTR, ranking] !")

        today = datetime.today()
        today_str = f"{today.year}-{today.month}-{today.day}"
        log_dir = os.path.join(self.base_dir, "log", f"{today_str}_{config['log_name']}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Fix watch_metric to match topk values
        if 'watch_metric' in config and 'topk' in config:
            current_watch_metric = config['watch_metric']
            # Extract the K value from watch_metric (e.g., 'ndcg@5' -> '5')
            if '@' in current_watch_metric:
                metric_name, metric_k = current_watch_metric.rsplit('@', 1)
                # Use the first topk value
                first_topk = config['topk'][0]
                config['watch_metric'] = f"{metric_name}@{first_topk}"
                if current_watch_metric != config['watch_metric']:
                    print(f"⚠ Updated watch_metric from '{current_watch_metric}' to '{config['watch_metric']}' to match topk={config['topk']}")

        early_stop_metrics = None
        if 'early_stop_metrics' in config:
            raw_metrics = config['early_stop_metrics']
            if isinstance(raw_metrics, str):
                early_stop_metrics = [m.strip() for m in raw_metrics.split(',') if m.strip()]
            else:
                early_stop_metrics = list(raw_metrics)
            if len(early_stop_metrics) == 0:
                early_stop_metrics = None

        early_stop_weights = None
        if early_stop_metrics is not None:
            raw_weights = config.get('early_stop_weights')
            if raw_weights is None:
                early_stop_weights = [1.0] * len(early_stop_metrics)
            elif isinstance(raw_weights, str):
                early_stop_weights = [float(w.strip()) for w in raw_weights.split(',') if w.strip()]
            else:
                early_stop_weights = list(raw_weights)
            if len(early_stop_weights) != len(early_stop_metrics):
                raise ValueError(
                    "early_stop_weights must have the same length as early_stop_metrics"
                )
            weight_sum = sum(early_stop_weights)
            if weight_sum <= 0:
                raise ValueError("early_stop_weights must sum to a positive value")
            early_stop_weights = [w / weight_sum for w in early_stop_weights]

        early_stop_patience = int(config.get('early_stop_patience', 0) or 0)
        early_stop_min_delta = float(config.get('early_stop_min_delta', 0.0) or 0.0)

        # Initialize wandb
        wandb_run = None
        use_wandb = bool(config.get('use_wandb', True))
        if use_wandb and WANDB_AVAILABLE:
            # Keep the W&B run name identical to the configured experiment name.
            # In particular, base runs may inherit a rank_model default (e.g. APR),
            # but that must not appear in their name when fair-rank is disabled.
            model_name = config.get('model', 'unknown')
            rank_model = config.get('rank_model', 'base') if config.get('fair-rank', False) else 'base'
            wandb_run_name = config.get('log_name') or f"{model_name}-{rank_model}"
            
            try:
                wandb_run = wandb.init(
                    project="fairdiverse-recommendation",
                    name=wandb_run_name,
                    config=config,
                    group=model_name,
                    tags=[rank_model, config.get('dataset', 'unknown')],
                    reinit=True
                )
                print(f"✓ Wandb run initialized: {wandb_run_name}")
            except Exception as exc:
                # Experiment tracking is optional; cluster-side service startup
                # failures must not abort model training.
                wandb_run = None
                print(f"⚠ Wandb initialization failed; continuing without it: {exc}")
        else:
            reason = "disabled by config" if not use_wandb else "wandb package unavailable"
            print(f"⚠ Wandb {reason}. Skipping wandb logging.")

        # Compare a locally generated native GRID checkpoint without applying a
        # second FairDiverse training loop. The canonical standalone metrics are
        # still produced by GRID/run_independent_pipeline.sh; this path checks
        # the FairDiverse adapter/evaluator on the same local model weights.
        if config.get('model') == 'GRID' and config.get('grid_eval_only', False):
            if config.get('fair-rank', False):
                raise ValueError("grid_eval_only cannot be combined with fair-rank")
            if not config.get('grid_checkpoint_path'):
                raise ValueError("grid_eval_only requires grid_checkpoint_path")
            self.Model.eval()
            # The primary result must use the same FairDiverse full-catalog
            # evaluator as the fair-ranking experiments.  Native GRID beam
            # metrics are an optional parity diagnostic and are deliberately
            # kept out of W&B so that baseline/fair-run charts remain directly
            # comparable.
            if config.get('store_scores', False):
                test_result, ranking_scores = evaluator.eval(
                    test_loader, self.Model, store_scores=True
                )
                save_npz(os.path.join(log_dir, 'ranking_scores.npz'), ranking_scores)
                save_npz(
                    os.path.join(log_dir, 'ground_truth.npz'),
                    evaluator.last_ground_truth,
                )
            else:
                test_result = evaluator.eval(test_loader, self.Model)
            native_result = None
            if config.get('grid_native_diagnostics', False):
                native_result = getattr(evaluator, 'last_grid_native_result', None)
                if native_result is None:
                    native_config = dict(config)
                    native_config['grid_native_metrics'] = True
                    native_evaluator = Ranking_Evaluator(native_config)
                    native_result = native_evaluator.eval(test_loader, self.Model)
                    native_evaluator.save_user_fairness(os.path.join(log_dir, 'user_fairness_native'))
            evaluator.save_user_fairness(os.path.join(log_dir, 'user_fairness'))
            with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
                yaml.dump(config, f)
            with open(os.path.join(log_dir, 'test_result.json'), 'w') as f:
                json.dump(test_result, f)
            if native_result is not None:
                with open(os.path.join(log_dir, 'grid_native_test_result.json'), 'w') as f:
                    json.dump(native_result, f)
            print(f"GRID FairDiverse full-catalog evaluation: {test_result}")
            if native_result is not None:
                print(
                    "GRID native beam diagnostic (local JSON only; not sent to W&B): "
                    f"{native_result}"
                )
            if wandb_run is not None:
                wandb.log({f"test/{k}": v for k, v in test_result.items()})
                wandb.finish()
            return test_result

        # Re-evaluate a previously trained non-GRID model without entering the
        # training loop.  This is primarily used to backfill ranking-score and
        # ground-truth artifacts for historical runs whose best_model.pth was
        # retained.  GRID has its own native-checkpoint evaluation path above.
        checkpoint_eval_path = config.get('checkpoint_eval_only_path')
        if checkpoint_eval_path:
            if config.get('model') == 'GRID':
                raise ValueError(
                    "GRID score export must use grid_eval_only and "
                    "grid_checkpoint_path"
                )
            if config.get('fair-rank', False):
                raise ValueError("checkpoint_eval_only_path requires fair-rank=false")
            checkpoint_eval_path = os.path.abspath(
                os.path.expanduser(str(checkpoint_eval_path))
            )
            if not os.path.isfile(checkpoint_eval_path):
                raise FileNotFoundError(
                    f"Missing evaluation checkpoint: {checkpoint_eval_path}"
                )

            state_dict = torch.load(
                checkpoint_eval_path,
                map_location=config['device'],
                weights_only=True,
            )
            self.Model.load_state_dict(state_dict, strict=True)
            self.Model.eval()
            if config.get('store_scores', False):
                test_result, ranking_scores = evaluator.eval(
                    test_loader, self.Model, store_scores=True
                )
                save_npz(os.path.join(log_dir, 'ranking_scores.npz'), ranking_scores)
                save_npz(
                    os.path.join(log_dir, 'ground_truth.npz'),
                    evaluator.last_ground_truth,
                )
            else:
                test_result = evaluator.eval(test_loader, self.Model)

            if hasattr(evaluator, 'save_user_fairness'):
                evaluator.save_user_fairness(os.path.join(log_dir, 'user_fairness'))
            with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
                yaml.dump(config, f)
            with open(os.path.join(log_dir, 'test_result.json'), 'w') as f:
                json.dump(test_result, f)
            print(
                "Checkpoint evaluation complete: "
                f"checkpoint={checkpoint_eval_path}, result={test_result}"
            )
            if wandb_run is not None:
                wandb.log({f"test/{k}": v for k, v in test_result.items()})
                wandb.finish()
            return test_result

        print("start to train...")

        best_result = -1
        best_stop_value = -1
        bad_eval_steps = 0
        train_batch_step = 0
        optimizer_step = 0
        last_validation_optimizer_step = -1
        max_steps = int(config.get('max_steps', 0) or 0)
        val_check_interval_batches = int(config.get('val_check_interval_batches', 0) or 0)
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if val_check_interval_batches < 0:
            raise ValueError("val_check_interval_batches must be non-negative")
        if val_check_interval_batches > 0:
            print(
                "Using step-based GRID schedule: "
                f"validate every {val_check_interval_batches} training batches, "
                f"max optimizer steps={max_steps or 'unlimited'}"
            )

        def run_validation(epoch):
            nonlocal best_result, best_stop_value, bad_eval_steps
            nonlocal last_validation_optimizer_step

            eval_result = evaluator.eval(valid_loader, self.Model)
            watch_eval_value = eval_result[config['watch_metric']]
            stop_eval_value = watch_eval_value
            if early_stop_metrics is not None:
                missing_metrics = [m for m in early_stop_metrics if m not in eval_result]
                if missing_metrics:
                    raise KeyError(
                        f"Missing early stop metrics in eval_result: {missing_metrics}"
                    )
                stop_eval_value = sum(
                    weight * eval_result[metric_name]
                    for metric_name, weight in zip(early_stop_metrics, early_stop_weights)
                )

            save_metric_value = stop_eval_value if early_stop_metrics is not None else watch_eval_value
            if save_metric_value >= best_result:
                best_result = save_metric_value
                torch.save(self.Model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            last_validation_optimizer_step = optimizer_step
            print(f"eval result: {eval_result}, best result: {best_result}")
            if early_stop_metrics is not None:
                print(f"early_stop_score: {stop_eval_value:.6f}")
            print()

            if wandb_run is not None:
                wandb_log_dict = {f"val/{k}": v for k, v in eval_result.items()}
                wandb_log_dict['val/best_result'] = best_result
                if early_stop_metrics is not None:
                    wandb_log_dict['val/early_stop_score'] = stop_eval_value
                wandb_log_dict['epoch'] = epoch
                wandb_log_dict['optimizer_step'] = optimizer_step
                wandb.log(wandb_log_dict)

            should_stop = False
            if early_stop_patience > 0:
                if stop_eval_value >= best_stop_value + early_stop_min_delta:
                    best_stop_value = stop_eval_value
                    bad_eval_steps = 0
                else:
                    bad_eval_steps += 1
                    if bad_eval_steps >= early_stop_patience:
                        print(
                            f"Early stopping at optimizer step {optimizer_step} "
                            f"(no improvement for {bad_eval_steps} validations)."
                        )
                        should_stop = True

            self.Model.train()
            return should_stop

        stop_training = False

        for epoch in trange(config['epoch']):


            total_loss = 0
            self.Model.train()

            if config['fair-rank'] == True:
                if 'update_epoch' in config and epoch % config['update_epoch'] == 0:
                    if config['rank_model'] == 'FairDual':
                        self.Fair_Ranker.reset_parameters(len(train_data_df))
                    else:
                        self.Fair_Ranker.reset_parameters()

            #for user_ids, item_ids, group_ids, label in train_loader:
            _n_train_batches = len(train_loader)
            for _bi, train_datas in enumerate(train_loader):

                if data_type == "point":
                    interaction = {"user_ids": train_datas[0].to(self.device), "history_ids": train_datas[1].to(self.device),
                                   "item_ids": train_datas[2].to(self.device),
                                   "group_ids": train_datas[3].to(self.device), "label": train_datas[4].to(self.device)}
                    #ids = {"user_ids": train_datas[0].to(self.device), "item_ids": train_datas[1].to(self.device)}
                elif data_type == "pair":
                    interaction = {"user_ids": train_datas[0].to(self.device), "history_ids": train_datas[1].to(self.device),
                                   "item_ids": train_datas[2].to(self.device), "neg_item_ids": train_datas[3].to(self.device),
                                   "group_ids": train_datas[4].to(self.device), "neg_group_ids": train_datas[5].to(self.device)
                                   }
                    #ids = {"user_ids": train_datas[0].to(self.device), "item_ids": train_datas[1].to(self.device)}
                else: ###squential format
                    interaction = {"user_ids": train_datas[0].to(self.device), "history_ids": train_datas[1].to(self.device),
                                   "item_ids": train_datas[2].to(self.device), "group_ids": train_datas[3].to(self.device), }

                feed_user_dict = {"user_ids": train_datas[0].to(self.device), "history_ids": train_datas[1].to(self.device)}
                feed_item_ids = train_datas[2].to(self.device)

                if _bi % _accum == 0:
                    optimizer.zero_grad()
                #loss = self.Model.compute_loss(interaction)

                if config['fair-rank'] == True:

                    if self.Fair_Ranker.fair_type == 're-weight':
                        loss = self.Model.compute_loss(interaction)

                        # Re-weighting rankers consume scores only as detached
                        # NumPy inputs; this forward pass is not part of the
                        # optimization objective.  Building a second autograd
                        # graph while ``loss`` still retains the first one can
                        # almost double GRID's peak memory (and OOM FairDual on
                        # a 48 GB GPU), so run the score-only pass in inference
                        # mode.
                        with torch.no_grad():
                            scores = self.Model(feed_user_dict, feed_item_ids)


                        input_dict = {'items': train_datas[2].detach().cpu().numpy(), 'loss': loss.detach().cpu().numpy(),
                                      'scores': scores.detach().cpu().numpy()}

                        if config['rank_model'] == 'FairDual':
                            ## FairDual needs sample some items
                            exposure_sample_num = config['exposure_sample_num']
                            item_sample_ids = random.sample(range(config['item_num']), exposure_sample_num)
                            item_sample_ids = torch.tensor(item_sample_ids, dtype=torch.long).to(self.device)

                            # FairDual uses this ranking only to choose exposed
                            # items for its NumPy reweighter below. The scores
                            # never contribute to the training loss, so keeping
                            # one autograd graph per sampled item needlessly
                            # retains hundreds of GRID forward passes on the
                            # GPU and can exhaust a 48 GB card.
                            with torch.no_grad():
                                _, indices = self.Model.full_ranking(
                                    feed_user_dict, item_sample_ids, k=config['s_k']
                                )

                            items = item_sample_ids[indices.detach().cpu().numpy()].cpu().numpy()
                            input_dict['sample_items'] = items

                        weight = self.Fair_Ranker.reweight(input_dict=input_dict)
                        #print(weight)
                        loss = torch.mean(torch.tensor(weight).to(self.device)*loss)

                    elif self.Fair_Ranker.fair_type == "sample":
                        neg_item_ids, adj = self.Fair_Ranker.sample(interaction, self.Model)
                        interaction['neg_item_ids'] = neg_item_ids
                        loss = self.Model.compute_loss(interaction)
                        if config['rank_model'] == "FairNeg":
                            group_loss = loss.detach().cpu().numpy()
                            group_loss = np.matmul(group_loss, adj)/(np.sum(adj, axis=0, keepdims=False)+1)
                            self.Fair_Ranker.accumulate_epoch_loss(group_loss)


                        loss = torch.mean(loss)

                    elif self.Fair_Ranker.fair_type == "regularizer":
                        loss = self.Model.compute_loss(interaction)
                        scores = self.Model(feed_user_dict, feed_item_ids)
                        input_dict = {'items': train_datas[2], 'loss': loss, 'scores':scores}

                        if config['rank_model'] == 'DPR':
                            input_dict = {'item_ids': train_datas[2].to(self.device), 'scores': scores,
                                          "neg_item_ids": train_datas[3].to(self.device),
                                          "group_ids": train_datas[4].to(self.device),
                                          "neg_group_ids": train_datas[5].to(self.device)
                                          }
                            fair_loss = self.Fair_Ranker.fairness_loss(input_dict, self.Model)
                            loss = torch.mean(loss) + fair_loss

                        else:
                            fair_loss = self.Fair_Ranker.fairness_loss(input_dict)
                            loss = torch.mean(loss) + config['fair_lambda'] * fair_loss

                    else: ##other type of fairness method, which modifies embeddings during the evaluation process
                        loss = self.Model.compute_loss(interaction)
                        loss = torch.mean(loss)

                else:
                    loss = self.Model.compute_loss(interaction)
                    loss = torch.mean(loss)

                (loss / _accum).backward()
                total_loss += loss.item()
                train_batch_step += 1
                if (_bi + 1) % _accum == 0 or _bi == _n_train_batches - 1:
                    optimizer.step()
                    optimizer_step += 1

                    if max_steps > 0 and optimizer_step >= max_steps:
                        print(f"Reached max_steps={max_steps}.")
                        stop_training = True

                if (
                    not stop_training
                    and val_check_interval_batches > 0
                    and train_batch_step % val_check_interval_batches == 0
                ):
                    stop_training = run_validation(epoch)

                if stop_training:
                    break



            if val_check_interval_batches == 0 and epoch % config['eval_step'] == 0:
                stop_training = run_validation(epoch)


            print("epoch: %d loss: %.3f" %(epoch, total_loss/ len(train_loader)))
            
            # Log training loss to wandb
            if wandb_run is not None:
                wandb.log({
                    'train/loss': total_loss / len(train_loader),
                    'epoch': epoch,
                    'optimizer_step': optimizer_step,
                })

            if stop_training:
                break


        best_model_path = os.path.join(log_dir, "best_model.pth")
        # A short step-bounded fine-tune can stop before its periodic validation
        # boundary. Evaluate the final weights, and ensure every training run
        # has a saved model before testing.
        if (
            config.get('grid_fair_finetune', False)
            and last_validation_optimizer_step != optimizer_step
        ) or not os.path.isfile(best_model_path):
            run_validation(config['epoch'])

        print(f"training complete! start to save the config and model...")
        print(f" config files are dump in {log_dir}")
        with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)

        print("start to testing...")
        self.Model.load_state_dict(torch.load(best_model_path))  # load state_dict
        self.Model.eval()  # change to eval model
        if config['store_scores'] == False:
            test_result = evaluator.eval(test_loader, self.Model)
        else:
            test_result, coo_matrix = evaluator.eval(test_loader, self.Model, store_scores=True)
            save_npz(os.path.join(log_dir, 'ranking_scores.npz'), coo_matrix) ##prepared for re-ranking stage
            save_npz(
                os.path.join(log_dir, 'ground_truth.npz'),
                evaluator.last_ground_truth,
            )


        if hasattr(evaluator, 'save_user_fairness'):
            evaluator.save_user_fairness(os.path.join(log_dir, 'user_fairness'))
        with open(os.path.join(log_dir, 'test_result.json'), 'w') as file:
            json.dump(test_result, file)
        print(test_result)
        print(f"dump in {log_dir}")
        
        # Log test results to wandb
        if wandb_run is not None:
            wandb_test_dict = {f"test/{k}": v for k, v in test_result.items()}
            wandb.log(wandb_test_dict)
            
            # Create a summary table for wandb
            summary_data = []
            for k in config.get('topk', [10, 20]):
                row = {
                    'K': k,
                    'NDCG': test_result.get(f'ndcg@{k}', 0),
                    'MRR': test_result.get(f'mrr@{k}', 0),
                    'HR': test_result.get(f'hr@{k}', 0),
                    'MMF': test_result.get(f'mmf@{k}', 0),
                    'GINI': test_result.get(f'gini@{k}', 0),
                    'Entropy': test_result.get(f'entropy@{k}', 0),
                }
                summary_data.append(row)
            
            # Log summary table
            wandb.log({"test/metrics_summary": wandb.Table(dataframe=pd.DataFrame(summary_data))})
            
            print("✓ Results logged to wandb")
            wandb.finish()
