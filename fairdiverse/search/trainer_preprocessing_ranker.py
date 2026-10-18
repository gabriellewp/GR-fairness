import os

import pandas as pd
import yaml
import wandb

from .datasets.COMPAS import Compas
from .preprocessing_model import fairness_method_mapping
from .ranker_model import ranker_mapping


class RankerTrainer(object):
    def __init__(self, train_config):
        self.train_config = train_config
        self.load_configs()
        
        # Initialize wandb
        preprocessing_model = train_config['train_ranker_config'].get('preprocessing_model', 'baseline')
        ranker = train_config['train_ranker_config'].get('ranker', 'RankNet')
        k_values = train_config['train_ranker_config']['evaluate'].get('k_list', [])
        k_str = f"K{'-'.join(map(str, k_values))}" if k_values else ""
        
        wandb.init(
            project="fairdiverse-search",
            name=f"{preprocessing_model}-{ranker}-{k_str}",
            config=train_config
        )

    def load_configs(self):
        # Load the configs for the dataset and for the pre-processing method
        with open(os.path.join(self.train_config['task'], "properties", "dataset",
                               self.train_config['dataset'].lower() + ".yaml"), 'r') as f:
            data_config = yaml.safe_load(f)
        self.train_config["data_reader_class"] = data_config["data_reader_class"]

        print(self.train_config['train_ranker_config']['preprocessing_model'])
        with open(os.path.join(self.train_config['task'], "properties", "models",
                               self.train_config['train_ranker_config']['preprocessing_model'] + ".yaml"),
                  'r') as f:
            model_config = yaml.safe_load(f)
        
        # Only set pre_processing_config if it exists in the model config
        # Baseline models (like RankNet) don't have preprocessing
        if "pre_processing_config" in model_config:
            self.train_config["pre_processing_config"] = model_config["pre_processing_config"]

        print(self.train_config)

    def train(self):
        # train the ranker with the pre-processing method
        if self.train_config["data_reader_class"]["name"] == "COMPAS":
            dataset = Compas(self.train_config["data_reader_class"])

        if "pre_processing_config" in self.train_config:
            fairness_method = fairness_method_mapping[self.train_config["pre_processing_config"]["name"]](
                self.train_config["pre_processing_config"], dataset)

        ranker = ranker_mapping[self.train_config['train_ranker_config']['name']](
            self.train_config['train_ranker_config'], dataset)

        for run in range(dataset.k_fold):
            data_train, data_test = dataset.read_data(run)

            # Create a combined dataframe with both original and fair columns
            if "pre_processing_config" in self.train_config:
                if self.train_config["pre_processing_config"]["qid_process"]:
                    data_train_fair = []
                    data_test_fair = []
                    qids = data_train[dataset.query_col].unique()
                    for qid in qids:
                        root_model_path = fairness_method.model_path.copy()
                        fairness_method.model_path = os.path.join(root_model_path, str(run), str(qid))

                        data_qid_train = data_train[data_train[dataset.query_col] == qid]
                        data_qid_test = data_test[data_test[dataset.query_col] == qid]

                        fairness_method.fit(data_qid_train, str(run))

                        data_qid_train_fair = fairness_method.transform(data_qid_train, str(run))
                        data_qid_test_fair = fairness_method.transform(data_qid_test, str(run))

                        data_train_fair.append(data_qid_train_fair)
                        data_test_fair.append(data_qid_test_fair)

                    data_train_fair = pd.concat(data_train_fair)
                    data_train_fair.to_csv(os.path.join(fairness_method.fair_data_path, 'fair_train_data.csv'))

                    data_test_fair = pd.concat(data_test_fair)
                    data_test_fair.to_csv(os.path.join(fairness_method.fair_data_path, 'fair_test_data.csv'))
                else:
                    fairness_method.fit(data_train, str(run))
                    data_train_fair = fairness_method.transform(data_train, str(run), "train")
                    data_test_fair = fairness_method.transform(data_test, str(run), "test")

                # The preprocessing methods already return data with BOTH original and fair columns
                # Fair columns have '_fair' suffix, original columns keep original names
                # So we can use the transformed data directly
                ranker_train_data = data_train_fair
                ranker_test_data = data_test_fair
            else:
                ranker_train_data = data_train
                ranker_test_data = data_test
                
            ranker.train(ranker_train_data, ranker_test_data, run)
            prediction = ranker.predict(ranker_test_data, run, "test")

            ranker.evaluate_run(prediction, run, "test")

        ranker.evaluate("test")  # computes mean over runs
        
        # Log metrics to wandb
        self._log_metrics_to_wandb(ranker)
        wandb.finish()
    
    def _log_metrics_to_wandb(self, ranker):
        """Read evaluation metrics from CSV files and log to wandb"""
        from pathlib import Path
        
        eval_path = Path(self.train_config['data_reader_class']['path']) / \
                    self.train_config['train_ranker_config']['preprocessing_model'] / "evaluate"
        
        if not eval_path.exists():
            return
        
        # Read aggregated metrics (Eval_QID_*_*.csv files in eval_path root)
        metrics_summary = {}
        for csv_file in eval_path.glob("Eval_QID_*_*.csv"):
            df = pd.read_csv(csv_file)
            metric_name = csv_file.stem.split('_')[-1]  # e.g., 'diversity', 'igf', 'individual'
            
            # Log metrics for each ranking type and k value
            for _, row in df.iterrows():
                rank_type = row.get('rank', 'unknown')
                k = row.get('k', 'unknown')
                group = row.get('group', 'all')
                
                # Create metric key
                key = f"{metric_name}/{rank_type}/k{k}/{group}"
                
                # Log all numeric columns except index columns
                for col in df.columns:
                    if col not in ['rank', 'group', 'k', 'Unnamed: 0'] and pd.api.types.is_numeric_dtype(df[col]):
                        value = row[col]
                        if not pd.isna(value):
                            metrics_summary[f"{key}/{col}"] = value
        
        # Log all metrics to wandb
        if metrics_summary:
            wandb.log(metrics_summary)
            print(f"Logged {len(metrics_summary)} metrics to wandb")
