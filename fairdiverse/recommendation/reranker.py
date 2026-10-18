import numpy as np
import os
import yaml
from scipy.sparse import save_npz, load_npz
from .rerank_model import CPFair, FairRec, FairRecPlus, k_neighbor, min_regularizer, PMMF, Welf, TaxRank, FairSync, RAIF, ElasticRank, TIGERFair
from .metric import dcg, NDCG, HR, MRR, MMF, Gini, Entropy, MinMaxRatio, EF
from datetime import datetime
import json

from .utils import Build_Adjecent_Matrix


class RecReRanker(object):
    def __init__(self, train_config):
        """Initialize In-processing and base models.

        :param train_config: Your custom config files.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset = train_config['dataset']
        #self.stage = stage
        self.train_config = train_config


    def load_configs(self, dir):
        """
           Loads and merges configuration files for the model, dataset, and evaluation.

           This function loads multiple YAML configuration files, including the process configuration,
           dataset-specific settings, model configurations, and evaluation parameters. All configurations
           are merged, with the highest priority given to the class's own `config` attribute.

           :param dir: The directory where the main process configuration file is located.
           :return: A dictionary containing the merged configuration from all files.
        """
        print("start to load config...")
        with open(os.path.join(dir, "process_config.yaml"), 'r') as f:
            config = yaml.safe_load(f)
        # print(train_data_df.head())

        if self.train_config['fair-rank'] == True:
            print("start to load model...")
            with open(os.path.join(self.base_dir, "properties", "models.yaml"), 'r') as f:
                model_config = yaml.safe_load(f)

            model_path = os.path.join(self.base_dir, "properties", "models", self.train_config['model'] + ".yaml")
            # if not os.path.exists(model_path):
            #     raise NotImplementedError("we do not support such model type!")
            with open(model_path, 'r') as f:
                model_config.update(yaml.safe_load(f))
            config.update(model_config)

        with open(os.path.join(self.base_dir, "properties", "evaluation.yaml"), 'r') as f:
            config.update(yaml.safe_load(f))

        config.update(self.train_config)  ###train_config has highest rights
        print("your loading config is:")
        print(config)

        return config

    def rerank(self, ranking_scores=None, ground_truth=None):
        """
            Training post-processing main workflow.
        """

        dir = os.path.join("recommendation", "processed_dataset", self.dataset)
        config = self.load_configs(dir)

        # Trainer writes log dirs to self.base_dir/log (fairdiverse/recommendation/log/).
        # For backwards compatibility also search recommendation/log/ (cwd-relative).
        candidate_base_dirs = [
            os.path.join(self.base_dir, "log"),
            os.path.join("recommendation", "log"),
        ]
        if ranking_scores is None:
            ranking_store_path = config['ranking_store_path']
            ranking_score_path = None
            for base_log_dir in candidate_base_dirs:
                if not os.path.exists(base_log_dir):
                    continue
                # Exact match first
                exact = os.path.join(base_log_dir, ranking_store_path)
                if os.path.exists(exact):
                    ranking_score_path = exact
                    break
                # Fuzzy match: log dirs are saved as "{date}_{log_name}", so search by suffix
                matches = sorted(
                    [d for d in os.listdir(base_log_dir)
                     if d == ranking_store_path or d.endswith('_' + ranking_store_path)],
                    reverse=True  # most recent date first
                )
                if matches:
                    ranking_score_path = os.path.join(base_log_dir, matches[0])
                    print(f"Resolved ranking_store_path '{ranking_store_path}' → '{ranking_score_path}'")
                    break
            if ranking_score_path is None:
                raise ValueError(f"Could not find log directory for '{ranking_store_path}' in any of: "
                                 f"{candidate_base_dirs}. "
                                 f"Run in-processing first to generate ranking_scores.npz, "
                                 f"then re-run post-processing.")
            print("loading ranking scores....")
            file = os.path.join(ranking_score_path, "ranking_scores.npz")
            ranking_scores = load_npz(file).toarray() #[user_num, item_num]
            ranking_scores[ranking_scores == 0] = -1000.0
            ground_truth_file = os.path.join(ranking_score_path, "ground_truth.npz")
            if os.path.isfile(ground_truth_file):
                ground_truth = load_npz(ground_truth_file).tocsr()
            else:
                print(
                    "WARNING: ground_truth.npz is missing; ground-truth NDCG, "
                    "HR, and MRR cannot be reported. Re-run the score export."
                )
        else:
            print("reusing ranking scores already loaded by TIGERFair sweep....")

        if ground_truth is not None and ground_truth.shape != ranking_scores.shape:
            raise ValueError(
                "ground_truth.npz and ranking_scores.npz must have identical "
                f"shapes; got {ground_truth.shape} and {ranking_scores.shape}"
            )

        if config.get('require_ground_truth', False) and ground_truth is None:
            raise FileNotFoundError(
                "This run requires ground_truth.npz, but the ranking-score export "
                "does not contain it. Re-run the score-export experiment before "
                "running the TIGERFair sweep."
            )

        fairness_modes = config.get('fairness_modes')
        lambda_values = config.get('lambda_values')
        if config.get('model') == 'TIGERFair' and (
            fairness_modes is not None or lambda_values is not None
        ):
            fairness_modes = fairness_modes or [config.get('fairness_mode', 'normalized_debt')]
            lambda_values = lambda_values or [config.get('lambda', 0.5)]
            allowed_modes = {
                'positive_deficit', 'cumulative_debt', 'normalized_debt'
            }
            unknown_modes = sorted(set(fairness_modes) - allowed_modes)
            if unknown_modes:
                raise ValueError(f"Unknown TIGERFair fairness modes: {unknown_modes}")

            sweep_results = {}
            base_log_name = config['log_name']
            for fairness_mode in fairness_modes:
                for fairness_lambda in lambda_values:
                    variant = dict(self.train_config)
                    variant.pop('fairness_modes', None)
                    variant.pop('lambda_values', None)
                    variant['fairness_mode'] = fairness_mode
                    variant['lambda'] = float(fairness_lambda)
                    lambda_name = format(float(fairness_lambda), 'g').replace('.', 'p')
                    variant['log_name'] = (
                        f"{base_log_name}-{fairness_mode.replace('_', '-')}-"
                        f"lambda-{lambda_name}"
                    )
                    print(
                        f"Running TIGERFair sweep variant: mode={fairness_mode}, "
                        f"lambda={fairness_lambda}"
                    )
                    result = RecReRanker(variant).rerank(
                        ranking_scores=ranking_scores,
                        ground_truth=ground_truth,
                    )
                    sweep_results[f"{fairness_mode}/lambda={fairness_lambda}"] = result

            relevance_budget = float(config.get('max_ndcg_loss', 0.05))
            sweep_summary = self._summarize_sweep(
                ranking_scores,
                ground_truth,
                config['topk'],
                sweep_results,
                relevance_budget,
            )

            today = datetime.today()
            today_str = f"{today.year}-{today.month}-{today.day}"
            sweep_dir = os.path.join(
                "fairdiverse", "recommendation", "log",
                f"{today_str}_{base_log_name}-sweep",
            )
            os.makedirs(sweep_dir, exist_ok=True)
            with open(os.path.join(sweep_dir, 'sweep_results.json'), 'w') as file:
                json.dump(sweep_results, file, indent=2)
            with open(os.path.join(sweep_dir, 'sweep_summary.json'), 'w') as file:
                json.dump(sweep_summary, file, indent=2)
            with open(os.path.join(sweep_dir, 'config.yaml'), 'w') as file:
                yaml.dump(config, file)
            print(f"TIGERFair sweep summary dumped in {sweep_dir}")
            return sweep_results

        ###we need to remove the group do not appear after the ranking phase, for evaluate full group, please evaluate in retrieval stage

        if config['fair-rank'] == False:
            config['model'] = None


        if config['model'] == "CPFair":
            Reranker = CPFair(config)
        elif config['model'] == "FairRec":
            Reranker = FairRec(config)
        elif config['model'] == "FairRecPlus":
            Reranker = FairRecPlus(config)
        elif config['model'] == 'k_neighbor':
            Reranker = k_neighbor(config)
        elif config['model'] == 'min_regularizer':
            Reranker = min_regularizer(config)
        elif config['model'] == 'PMMF':
            Reranker = PMMF(config)
        elif config['model'] == 'Welf':
            Reranker = Welf(config)
        elif config['model'] == 'TaxRank':
            Reranker = TaxRank(config)
        elif config['model'] == 'FairSync':
            Reranker = FairSync(config)
        elif config['model'] == 'RAIF':
            Reranker = RAIF(config)
        elif config['model'] == 'ElasticRank':
            Reranker = ElasticRank(config)
        elif config['model'] == 'TIGERFair':
            Reranker = TIGERFair(config)
        elif config['model'] == None:
            print("No model loaded!")
        else:
            raise NotImplementedError(f"We do not support the model type {self.train_config['model']}")

        #item_scores = np.sum(ranking_scores, axis=0, keepdims=False)

        metrics = ["ranking_retention_ndcg", "u_loss"]
        if ground_truth is not None:
            metrics.extend([
                "ground_truth_ndcg",
                "ground_truth_hr",
                "ground_truth_mrr",
            ])
        rerank_result = {}
        exposure_result = {}
        for k in config['topk']:
            rerank_result.update({f"{m}@{k}":0 for m in metrics})

            if config['fair-rank'] == True:
                rerank_list = Reranker.rerank(ranking_scores, k)
            else:
                result_item = np.argsort(ranking_scores,axis=-1)[:,::-1]
                rerank_list = result_item[:,:k]

            # Keep the standard FairDiverse count-based exposure metrics for
            # direct comparison with in-processing baselines.  TIGERFair can
            # additionally report position-aware exposure (uppercase keys),
            # which reflects the discounted exposure used by its objective.
            standard_exposure_list = np.zeros(config['group_num'])
            exposure_list = np.zeros(config['group_num'])
            for u in range(len(rerank_list)):
                sorted_result_score = np.sort(ranking_scores[u])[::-1]
                true_dcg = dcg(sorted_result_score, k)
                rerank_items = rerank_list[u]

                for rank, i in enumerate(rerank_items):
                    if config['fair-rank'] == False:
                        _, iid2pid = Build_Adjecent_Matrix(config)
                    else:
                        iid2pid = Reranker.iid2pid
                    if i not in iid2pid.keys():
                        gid = 0
                    else:
                        gid = iid2pid[i]
                    if config['fairness_type'] == "Exposure":
                        standard_exposure_list[gid] += 1
                        if config.get('position_aware_exposure', False):
                            exposure_list[gid] += 1.0 / np.log2(rank + 2.0)
                        else:
                            exposure_list[gid] += 1
                    else:
                        exposure_list[gid] += np.round(ranking_scores[u][i], config['decimals'])
                reranked_score = ranking_scores[u][rerank_items]
                # FIXED: Use ranking order, not sorted scores
                # OLD (inflated NDCG): pre_dcg = dcg(np.sort(reranked_score)[::-1], k)
                pre_dcg = dcg(reranked_score, k)
                if true_dcg > 0:
                    rerank_result[f"ranking_retention_ndcg@{k}"] += pre_dcg/true_dcg
                rerank_result[f"u_loss@{k}"] += (np.sum(sorted_result_score[:k]) - np.sum(reranked_score[:k]))/k

                if ground_truth is not None:
                    positive_items = ground_truth.getrow(u).indices
                    relevance = np.isin(rerank_items, positive_items).astype(np.int8)
                    ideal_relevance = np.ones(len(positive_items), dtype=np.int8)
                    rerank_result[f"ground_truth_ndcg@{k}"] += NDCG(
                        relevance, ideal_relevance, k
                    )
                    rerank_result[f"ground_truth_hr@{k}"] += HR(
                        relevance, ideal_relevance, k
                    )
                    rerank_result[f"ground_truth_mrr@{k}"] += MRR(relevance, k)

            for metric in metrics:
                rerank_result[f"{metric}@{k}"] /= len(rerank_list)

            # Standard result schema used by the ordinary FairDiverse
            # baselines.  Keep the explicit ground_truth_* names too so older
            # TIGERFair result consumers continue to work.
            if ground_truth is not None:
                rerank_result[f"ndcg@{k}"] = rerank_result[
                    f"ground_truth_ndcg@{k}"
                ]
                rerank_result[f"hr@{k}"] = rerank_result[
                    f"ground_truth_hr@{k}"
                ]
                rerank_result[f"mrr@{k}"] = rerank_result[
                    f"ground_truth_mrr@{k}"
                ]
            rerank_result[f"mmf@{k}"] = MMF(
                standard_exposure_list,
                ratio=config.get('mmf_eval_ratio', 0.2),
            )
            rerank_result[f"gini@{k}"] = Gini(standard_exposure_list)
            rerank_result[f"entropy@{k}"] = Entropy(standard_exposure_list)
            rerank_result[f"maxminratio@{k}"] = MinMaxRatio(
                standard_exposure_list
            )
            for fairness_metric in self.train_config['fairness_metrics']:
                if fairness_metric == 'MinMaxRatio':
                    rerank_result[f"MinMaxRatio@{k}"] = MinMaxRatio(exposure_list)
                elif fairness_metric == 'MMF':
                    rerank_result[f"MMF@{k}"] = MMF(exposure_list)
                elif fairness_metric == 'Entropy':
                    rerank_result[f"Entropy@{k}"] = Entropy(exposure_list)
                elif fairness_metric == 'GINI':
                    rerank_result[f"GINI@{k}"] = Gini(exposure_list)
                elif fairness_metric == 'EF':
                    rerank_result[f"EF@{k}"] = EF(exposure_list)

            if config['model'] == 'TIGERFair':
                rerank_result[f"candidate_group_coverage@{k}"] = (
                    Reranker.last_candidate_group_coverage
                )

            #rerank_result[f"mmf@{k}"] = MMF(exposure_list)
            #rerank_result[f"gini@{k}"] = Gini(exposure_list)
            #print(exposure_list)
            exposure_result[f"top@{k}"] = str(list(exposure_list))


        for k in rerank_result.keys():
            rerank_result[k] = np.round(rerank_result[k], config['decimals'])


        today = datetime.today()
        today_str = f"{today.year}-{today.month}-{today.day}"
        log_dir = os.path.join("fairdiverse", "recommendation", "log", f"{today_str}_{config['log_name']}")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'test_result.json'), 'w') as file:
            json.dump(rerank_result, file)
        with open(os.path.join(log_dir, 'exposure_result.json'), 'w') as file:
            json.dump(exposure_result, file)
            #file.write(str(exposure_list))
        #print("exposure list:")
        #print(exposure_list)
        print(rerank_result)

        with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)

        print(f"result and config dump in {log_dir}")

        return rerank_result

    @staticmethod
    def _summarize_sweep(ranking_scores, ground_truth, topk, results, max_loss):
        """Compare reranked relevance with the unmodified candidate ordering."""
        if ground_truth is None:
            return {
                "status": "ground_truth_missing",
                "max_ndcg_loss": max_loss,
                "variants": {},
            }

        baseline_order = np.argsort(ranking_scores, axis=-1)[:, ::-1]
        baseline = {}
        for k in topk:
            total = 0.0
            for user, items in enumerate(baseline_order[:, :k]):
                positive_items = ground_truth.getrow(user).indices
                relevance = np.isin(items, positive_items).astype(np.int8)
                total += NDCG(
                    relevance,
                    np.ones(len(positive_items), dtype=np.int8),
                    k,
                )
            baseline[f"ground_truth_ndcg@{k}"] = np.round(
                total / len(baseline_order), 4
            )

        variants = {}
        for name, result in results.items():
            losses = {}
            eligible = True
            for k in topk:
                metric = f"ground_truth_ndcg@{k}"
                baseline_value = float(baseline[metric])
                reranked_value = float(result[metric])
                relative_loss = (
                    (baseline_value - reranked_value) / baseline_value
                    if baseline_value > 0 else 0.0
                )
                losses[f"relative_ndcg_loss@{k}"] = round(relative_loss, 4)
                eligible = eligible and relative_loss <= max_loss
            variants[name] = {
                **losses,
                "meets_relevance_budget": bool(eligible),
            }

        return {
            "status": "ok",
            "max_ndcg_loss": max_loss,
            "baseline": baseline,
            "variants": variants,
            "eligible_variants": [
                name for name, values in variants.items()
                if values["meets_relevance_budget"]
            ],
        }
