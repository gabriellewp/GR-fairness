import torch
from tqdm import tqdm, trange
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from .metric import NDCG, HR, MRR, AUC_score, MMF, Gini, Entropy
import os
import json
from .utils import Build_Adjecent_Matrix
from .metric import *
from .user_fairness import load_activity, ranking_metrics, save_report


def _legacy_topk_args(scores, k):
    """Match np.argsort(scores)[::-1][:k], including its unstable tie order."""
    k = min(k, len(scores))
    if k < 1:
        return np.empty(0, dtype=np.int64)
    if k == len(scores) or not np.isfinite(scores).all():
        return np.argsort(scores)[::-1][:k]
    candidates = np.argpartition(scores, len(scores) - k)[-k:]
    candidates = candidates[np.argsort(scores[candidates])[::-1]]
    values = scores[candidates]
    # Sorting a subset cannot reproduce quicksort's catalog-wide tie order.
    # Fall back for ties within top-K or across the cutoff, including ties
    # introduced by the evaluator's float32 score + 1 transformation.
    if np.any(values[:-1] == values[1:]) or np.count_nonzero(scores == values[-1]) > 1:
        return np.argsort(scores)[::-1][:k]
    return candidates




class Abstract_Evaluator(object):
    def __init__(self, config):
        self.config = config
        self.M, self.iid2pid = Build_Adjecent_Matrix(config)


    def eval(self, dataloader, model, store_scores = False):
        """
            Evaluates the model on the provided dataloader and calculates performance metrics.


            :param dataloader: The data loader that provides batches of user-item interactions and corresponding labels.
            :param model: The model to evaluate.
            :param store_scores: Whether to return the predicted scores as a sparse matrix. Defaults to `False`.
            :return: A dictionary containing the evaluation metric(s) (e.g., AUC score).
        """
        pass


class CTR_Evaluator(Abstract_Evaluator):
    def __int__(self,config):
        super().__init__(config=config)

    def eval(self, dataloader, model, store_scores = False):
        """
            Evaluates the model on the provided dataloader and calculates performance metrics.

            This function runs the evaluation on a dataset using the provided model. It calculates the AUC score based on
            the predicted scores and ground truth labels. If `store_scores` is set to `True`, it also returns the evaluation
            results as a sparse matrix of predicted scores.

            :param dataloader: The data loader that provides batches of user-item interactions and corresponding labels.
            :param model: The model to evaluate.
            :param store_scores: Whether to return the predicted scores as a sparse matrix. Defaults to `False`.
            :return: A dictionary containing the evaluation metric(s) (e.g., AUC score), and optionally, a sparse matrix
                     of predicted scores.
        """
        model.eval()
        y_scores = []
        y_true = []

        row = []
        col = []
        data = []
        with torch.no_grad():
            for user_ids, item_ids, group_ids, label in tqdm(dataloader):
                row.extend(user_ids.numpy().tolist())
                col.extend(item_ids.numpy().tolist())
                user_ids, item_ids = user_ids.to(self.config['device']), item_ids.to(self.config['device'])

                score = model(user_ids, item_ids).cpu().numpy().tolist()
                data.extend(score)
                label = label.cpu().numpy().tolist()
                y_scores.extend(score)
                y_true.extend(label)

        auc_score = AUC_score(y_scores=y_scores, y_true=y_true)
        result_dict = {}
        result_dict["auc"] = np.round(auc_score,self.config['decimals'])

        if store_scores == False:
            return result_dict
        else:
            coo = coo_matrix((data, (row, col)), shape=(self.config['user_num'], self.config['item_num']))
            csr = coo.tocsr() #to remove the zero rows
            csr_eliminated = csr[csr.getnnz(1) > 0]
            coo = csr_eliminated.tocoo()
            return result_dict, coo


class Ranking_Evaluator(Abstract_Evaluator):
    def _start_user_fairness(self):
        self.last_user_metrics = []
        self._activity = None
        if self.config.get("user_fairness", False):
            path = self.config.get("user_activity_path")
            if not path:
                raise ValueError("user_fairness requires user_activity_path; run user_fairness prepare first")
            self._activity = load_activity(path)

    def _record_user_metrics(self, user, request, metrics):
        user = str(user)
        if user not in self._activity:
            raise ValueError(f"Missing activity mapping for user {user}")
        self.last_user_metrics.append({"user_id": user, "request_id": str(request), **metrics})

    def save_user_fairness(self, output_dir, split="test"):
        """Explicit saving avoids validation epochs overwriting test artifacts."""
        if not self.config.get("user_fairness", False):
            return None
        return save_report(
            self.last_user_metrics, self.config["user_activity_path"], output_dir,
            self.config.get("user_fairness_bootstrap_samples", 1000),
            self.config.get("seed", 42),
            {"split": split, "model": self.config.get("model"),
             "protocol": "native_beams" if self.config.get("grid_native_metrics") else "candidate_ranking",
             "topk": self.config["topk"],
             "settings": {key: self.config[key] for key in (
                 "dataset", "seed", "eval_split_method", "sample_num", "top_k_for_generation",
                 "grid_checkpoint_path", "checkpoint_eval_only_path", "fair-rank", "rank_model",
             ) if key in self.config}},
        )

    def __int__(self,config):
        super().__init__(config=config)

    def eval(self, dataloader, model, store_scores = False):
        """
            Evaluates the model on the provided dataloader and calculates performance metrics.

            This function runs the evaluation on a dataset using the provided model. It calculates the Ranking metrics based on
            the predicted scores and ground truth labels. If `store_scores` is set to `True`, it also returns the evaluation
            results as a sparse matrix of predicted scores.

            :param dataloader: The data loader that provides batches of user-item interactions and corresponding labels.
            :param model: The model to evaluate.
            :param store_scores: Whether to return the predicted scores as a sparse matrix. Defaults to `False`.
            :return: A dictionary containing the evaluation metric(s), and optionally, a sparse matrix
                     of predicted scores.
        """
        self._start_user_fairness()
        if (
            self.config.get("model") == "GRID"
            and self.config.get("grid_native_metrics", False)
        ):
            if store_scores:
                raise ValueError("grid_native_metrics does not produce a candidate score matrix")
            return self._eval_grid_native(dataloader, model)

        model.eval()
        generation_batch_size = int(self.config.get("grid_eval_generation_batch_size", 0))
        batched_grid = self.config.get("model") == "GRID" and generation_batch_size > 0
        if batched_grid and not hasattr(model, "full_predict_batch"):
            raise ValueError("GRID batching requires full_predict_batch")
        classic_batch_size = int(self.config.get("classic_eval_batch_size", 0))
        if classic_batch_size < 0:
            raise ValueError("classic_eval_batch_size must be nonnegative")
        batched_classic = self.config.get("model") == "SASRec" and classic_batch_size > 0
        if batched_classic and not hasattr(model, "full_predict_batch"):
            raise ValueError("SASRec batching requires full_predict_batch")
        self.last_grid_native_result = None
        reuse_native = bool(
            batched_grid and self.config.get("grid_native_diagnostics", False)
            and not self.config.get("user_fairness", False)
        )
        native_totals = {f"{metric}@{k}": 0.0 for metric in ("ndcg", "mrr", "hr")
                         for k in self.config['topk']}
        native_count = 0
        if batched_grid:
            print(f"[GRID] Batched evaluation: {generation_batch_size} users per generation; "
                  "indexed SID matching; original catalog ordering retained")
        if batched_classic:
            print(f"[SASRec] Batched evaluation: {classic_batch_size} users; "
                  "full catalog, legacy sigmoid scores and tie ordering; top-K metrics")
        y_scores = []
        y_true = []

        result_dict = {f"ndcg@{k}":0 for k in self.config['topk']}
        result_dict.update({f"mrr@{k}":0 for k in self.config['topk']})
        result_dict.update({f"hr@{k}":0 for k in self.config['topk']})
        result_dict.update({f"mmf@{k}": 0 for k in self.config['topk']})
        result_dict.update({f"gini@{k}": 0 for k in self.config['topk']})
        result_dict.update({f"entropy@{k}": 0 for k in self.config['topk']})
        result_dict.update({f"maxminratio@{k}": 0 for k in self.config['topk']})
        beam_diagnostics_enabled = bool(
            self.config.get("beam_diagnostics", False)
            and self.config.get("use_beam_search_eval", False)
        )
        if beam_diagnostics_enabled:
            num_levels = int(self.config["num_hierarchies"])
            result_dict["beam_target_coverage"] = 0
            result_dict.update({
                f"beam_prefix_coverage@{level}": 0
                for level in range(1, num_levels + 1)
            })
        exposure_dict = {f"top@{k}":np.zeros(self.config['group_num']) for k in self.config['topk']}
        index = 0

        #UI_matrix = np.zeros((self.config['user_num'], self.config['item_num']))
        # Keep stored rankings in CSR-ready, typed chunks.  The old code put
        # every row id, item id and score into Python lists.  Full-catalog
        # evaluation can contain hundreds of millions of entries, for which
        # the Python-object overhead alone consumes tens of GB and causes a
        # steadily growing host-memory OOM.  Evaluation visits rows in order,
        # so CSR only needs one compact indptr entry per user; row coordinates
        # do not need to be materialized at all.
        score_col_chunks = []
        score_data_chunks = []
        score_indptr = [0]
        ground_truth_row = []
        ground_truth_col = []
        compact_grid_scores = bool(
            store_scores
            and self.config.get("model") == "GRID"
            and self.config.get("grid_compact_ranking", False)
        )

        with torch.no_grad():

            #for user_ids, history_behavior, items, pos_length in tqdm(dataloader):
            for eval_data in tqdm(dataloader):
                user_ids, history_behavior, items, pos_length = eval_data
                batch_size, sample_size = items.shape #item
                #print(items.shape)
                #exit(0)
                pos_length = pos_length.cpu().numpy()
                batch_scores = None
                if batched_grid:
                    batch_scores = model.full_predict_batch(
                        {"user_ids": user_ids.to(self.config['device']),
                         "history_ids": history_behavior.to(self.config['device'])},
                        items, generation_batch_size,
                    ).numpy()
                elif batched_classic:
                    batch_scores = model.full_predict_batch(
                        {"user_ids": user_ids.to(self.config['device']),
                         "history_ids": history_behavior.to(self.config['device'])},
                        items, classic_batch_size,
                    ).numpy()

                for b in range(batch_size):
                    row_index = index
                    index = index + 1
                    real_item_ids = items[b].numpy()
                    if not (batched_grid or batched_classic):
                        real_item_ids = real_item_ids.tolist()
                    if store_scores:
                        positive_count = int(pos_length[b])
                        positive_items = real_item_ids[:positive_count]
                        ground_truth_row.extend([row_index] * positive_count)
                        ground_truth_col.extend(positive_items)
                    # print(row)
                    # print(col)
                    # has_duplicates = len(col) != len(set(col))
                    # print("是否有重复元素:", has_duplicates)
                    # exit(0)
                    #print(model.IR_type)
                    if batch_scores is not None:
                        score = batch_scores[b]
                        if beam_diagnostics_enabled:
                            beam_stats = model.beam_target_diagnostics(
                                items[b, :int(pos_length[b])].to(self.config['device']), b
                            )
                            for metric_name, metric_value in beam_stats.items():
                                result_dict[metric_name] += metric_value
                        if reuse_native and int(pos_length[b]) > 0:
                            targets = model._target_sids(
                                items[b, :int(pos_length[b])].to(self.config['device'])
                            ).cpu()
                            beams = model._last_generated_sids[b]
                            positions = (targets[:, None, :] == beams[None, :, :]).all(-1).nonzero()
                            rank = int(positions[:, 1].min()) + 1 if positions.numel() else None
                            for k in self.config['topk']:
                                if rank is not None and rank <= k:
                                    native_totals[f"ndcg@{k}"] += 1.0 / np.log2(rank + 1)
                                    native_totals[f"mrr@{k}"] += 1.0 / rank
                                    native_totals[f"hr@{k}"] += 1.0
                            native_count += 1
                    elif 'retrieval' not in model.IR_type:
                        #if self.config['data_type'] == 'point' or self.config['data_type'] == 'pair':
                        repeat_user_tensor = user_ids[b].repeat(sample_size).unsqueeze(0).to(self.config['device'])
                        #else:
                        repeat_history_tensor = history_behavior[b].repeat(sample_size, 1).unsqueeze(0).to(self.config['device'])

                        user_dict = {"user_ids": repeat_user_tensor,
                                     "history_ids": repeat_history_tensor}
                        i = items[b].to(self.config['device'])
                        score = model(user_dict, i.unsqueeze(0)).cpu().numpy()[0]

                    else:
                        user_dict = {"user_ids":user_ids[b].unsqueeze(0).to(self.config['device']),
                                     "history_ids":history_behavior[b].unsqueeze(0).to(self.config['device'])}
                        i = items[b].to(self.config['device'])

                        score = model.full_predict(user_dict, i.unsqueeze(0)).cpu().numpy()[0]
                        if beam_diagnostics_enabled:
                            positive_items = i[:int(pos_length[b])]
                            beam_stats = model.beam_target_diagnostics(positive_items)
                            for metric_name, metric_value in beam_stats.items():
                                result_dict[metric_name] += metric_value
                    raw_score = score
                    score = raw_score + 1.0
                    if store_scores:
                        if compact_grid_scores:
                            generated_mask = raw_score > 0
                        else:
                            generated_mask = np.ones(sample_size, dtype=bool)
                        stored_items = np.asarray(real_item_ids)[generated_mask]
                        score_col_chunks.append(stored_items.astype(np.int32, copy=False))
                        score_data_chunks.append(
                            score[generated_mask].astype(np.float32, copy=False)
                        )
                        score_indptr.append(score_indptr[-1] + len(stored_items))
                    # print(score)
                    # print(score.shape)
                    #ranked_score = np.sort(score)[::-1]
                    if batched_classic:
                        ranked_args = _legacy_topk_args(score, max(self.config['topk']))
                        rank_list = (ranked_args < int(pos_length[b])).astype(np.int8)
                        label_list = np.ones(min(int(pos_length[b]), max(self.config['topk'])))
                    elif batched_grid:
                        label_list = np.zeros(sample_size, dtype=np.int8)
                        label_list[:int(pos_length[b])] = 1
                    else:
                        label_list = [1] * pos_length[b] + [0] * (sample_size - pos_length[b])
                        label_list = np.array(label_list)
                    if not batched_classic:
                        ranked_args = np.argsort(score)[::-1]
                        rank_list = label_list[ranked_args]
                    # The ideal labels are binary positives followed by zeros.
                    # Metrics inspect only top-K: avoid sorting the entire
                    # catalog again inside NDCG and HR for every cutoff.
                    metric_labels = (
                        np.ones(min(int(pos_length[b]), max(self.config['topk'])))
                        if (batched_grid or batched_classic) else label_list
                    )
                    if self._activity is not None:
                        self._record_user_metrics(
                            user_ids[b].item(), row_index,
                            ranking_metrics(np.asarray(real_item_ids)[ranked_args].tolist(),
                                            real_item_ids[:int(pos_length[b])], self.config['topk']),
                        )
                    for k in self.config['topk']:
                        result_dict[f"ndcg@{k}"] += NDCG(rank_list, metric_labels, k)
                        result_dict[f"mrr@{k}"] += MRR(rank_list, k)
                        result_dict[f"hr@{k}"] += HR(rank_list, metric_labels, k)

                        ######count the exposures for the computing fairness degree#############
                        ids = ranked_args[:k]
                        rank_items = np.asarray(real_item_ids)[ids]
                        #print(rank_items)
                        for i, iid in enumerate(rank_items):
                            group_id = self.iid2pid[iid]
                            if self.config['fairness_type'] == "Exposure":
                                exposure_dict[f"top@{k}"][group_id] += 1
                            else:
                                exposure_dict[f"top@{k}"][group_id] += np.round(score[ids[i]], self.config['decimals'])
                            #exposure_dict[f"top@{k}"][group_id] += 1




        if reuse_native:
            if native_count == 0:
                raise ValueError("GRID native diagnostic received no positive targets")
            self.last_grid_native_result = {
                key: np.round(value / native_count, self.config['decimals'])
                for key, value in native_totals.items()
            }
            print(f"[GRID] Reused generated beams for native metrics over {native_count} targets")

        for k in self.config['topk']:
            #print(exposure_dict[f"top@{k}"])
            result_dict[f"mmf@{k}"] = MMF(exposure_dict[f"top@{k}"], ratio=self.config['mmf_eval_ratio']) * index
            result_dict[f"gini@{k}"] = Gini(exposure_dict[f"top@{k}"]) * index
            result_dict[f"entropy@{k}"] = Entropy(exposure_dict[f"top@{k}"]) * index
            result_dict[f"maxminratio@{k}"] = MinMaxRatio(exposure_dict[f"top@{k}"]) * index


        for key in result_dict.keys():
            result_dict[key] = np.round(result_dict[key]/index, self.config['decimals'])

        if store_scores == False:
            return result_dict
        else:
            # Keep held-out relevance aligned with the score-matrix row index.
            # A separate artifact is necessary because ranking_scores.npz only
            # contains model scores and cannot encode zero-scored positives.
            self.last_ground_truth = coo_matrix(
                (
                    np.ones(len(ground_truth_row), dtype=np.int8),
                    (ground_truth_row, ground_truth_col),
                ),
                shape=(index, self.config['item_num']),
            )
            if score_data_chunks:
                score_data = np.concatenate(score_data_chunks)
                score_data_chunks.clear()
                score_cols = np.concatenate(score_col_chunks)
                score_col_chunks.clear()
            else:
                score_data = np.empty(0, dtype=np.float32)
                score_cols = np.empty(0, dtype=np.int32)
            score_matrix = csr_matrix(
                (
                    score_data,
                    score_cols,
                    np.asarray(score_indptr, dtype=np.int64),
                ),
                shape=(index, self.config['item_num']),
                copy=False,
            )
            return result_dict, score_matrix

    def _eval_grid_native(self, dataloader, model):
        """Evaluate LOO targets at their rank in GRID's native generated beams."""
        self._start_user_fairness()
        model.eval()
        totals = {f"ndcg@{k}": 0.0 for k in self.config['topk']}
        totals.update({f"mrr@{k}": 0.0 for k in self.config['topk']})
        totals.update({f"hr@{k}": 0.0 for k in self.config['topk']})
        count = 0

        with torch.no_grad():
            for user_ids, histories, items, pos_lengths in tqdm(dataloader):
                for batch_index in range(items.shape[0]):
                    positive_count = int(pos_lengths[batch_index])
                    if self._activity is not None and positive_count != 1:
                        raise ValueError("Native GRID user fairness requires exactly one target per request")
                    if positive_count <= 0:
                        continue
                    history = histories[batch_index].unsqueeze(0).to(self.config['device'])
                    user = user_ids[batch_index].unsqueeze(0).to(self.config['device'])
                    positives = items[batch_index, :positive_count].to(self.config['device'])
                    generated, _ = model.generate(history, user)
                    target_sids = model._target_sids(positives)
                    matches = (
                        target_sids[:, None, :] == generated[0][None, :, :]
                    ).all(dim=-1)
                    matched_positions = matches.nonzero(as_tuple=False)
                    rank = None
                    if matched_positions.numel() > 0:
                        rank = int(matched_positions[:, 1].min().item()) + 1

                    for k in self.config['topk']:
                        if rank is not None and rank <= k:
                            totals[f"ndcg@{k}"] += 1.0 / np.log2(rank + 1)
                            totals[f"mrr@{k}"] += 1.0 / rank
                            totals[f"hr@{k}"] += 1.0
                    if self._activity is not None:
                        metrics = {}
                        for k in self.config['topk']:
                            hit = rank is not None and rank <= k
                            metrics[f"ndcg@{k}"] = float(1.0 / np.log2(rank + 1)) if hit else 0.0
                            metrics[f"recall@{k}"] = float(hit)
                        self._record_user_metrics(user_ids[batch_index].item(), count, metrics)
                    count += 1

        if count == 0:
            raise ValueError("GRID native evaluation received no positive targets")
        result = {
            key: np.round(value / count, self.config['decimals'])
            for key, value in totals.items()
        }
        print(f"[GRID] Native beam metrics over {count} leave-one-out targets")
        return result



class LLM_Evaluator(Abstract_Evaluator):
    def __init__(self, config):
        super().__init__(config=config)
        self.topk_list = config['topk']

    def get_data(self, data):
        """
        This method processes the input data to extract prediction lists, label lists, and score lists for each user.

        :param 'predict_list': A list of predicted items.
        :param 'positive_items': A list of items that are considered positive (e.g., liked or preferred by the user).
        :param 'scores': A list of scores corresponding to the predicted items, indicating the confidence of the prediction.

        :return:
        - `predict_lists`: A list of predict lists for all users.
        - `label_lists`: For each user, a list of binary labels indicating whether each predicted item is positive (1) or not (0).
        - `score_lists`: A list of score lists corresponding to the predicted items for all users.
        """
        # ground_truths = [i['positive_items'] for i in data]
        # sens_feat = [i['sensitiveAttribute'] for i in data]
        label_lists = []
        # ranking_lists = []
        score_lists = []
        predict_lists = []
        for user in data:
            p = user['predict_list']
            predict_lists.append(p)
            label_list = [1 if m in user['positive_items'] else 0 for m in p]
            # label_list = [1 if m in user['positive_items'] else 0 for m in user['item_candidates']]
            score = user['scores']
            score_lists.append(score)
            label_lists.append(label_list)
            # ranking_lists.append(ranking_list)

        return predict_lists, label_lists, score_lists

    def get_cates_value(self, iid2pid, predict, topk):
        """
        Get the category values based on predicted indices and their corresponding categories.

        This method processes the predicted indices along with their mapping to category IDs
        and returns a list of counts for each category, representing the frequency of occurrence
        in the top-k predictions.

        :param iid2pid : dict
            A dictionary mapping item indices (int) to their respective category IDs (int).
            If an item index is not found in the dictionary, it defaults to -1.

        :param predict : List[List[int]]
            A 2D list where each sublist contains the predicted indices (top-k predictions)
            for corresponding input data points.

        :param topk : int
            The number of top predictions considered for each data point. This determines
            how many elements from the beginning of each sublist in `predict` are processed.

        :return: List[int]
            A list of integers where each value corresponds to the count of occurrences for
            a specific category across all top-k predictions. The order of these counts matches
            the sorted order of category IDs as returned by `get_categories(iid2pid)`.
        """
        cates_name = self.get_categories(iid2pid)
        predict = [i[:topk] for i in predict]
        from collections import defaultdict
        cates_count = defaultdict(int)
        for p in predict:
            for prediction in p:
                c = iid2pid.get(prediction, -1)
                cates_count[c] += 1  # not score-based scores[idx][k]
        values = [cates_count[i] for i in cates_name]
        return values

    def cal_acc_score(self, label_lists, score_lists, topk):
        """
        Calculate accuracy scores for recommendation system evaluation.

        This method computes the average NDCG (Normalized Discounted Cumulative Gain), HR (Hit Ratio), and MRR (Mean Reciprocal Rank)
        at a specified `topk` cutoff for a list of ground-truth labels and corresponding prediction scores.

        :param label_lists : List[List[int]]
            A list of lists containing ground-truth labels. Each sublist represents the relevant items for a user or query.
        :param score_lists : List[List[float]]
            A list of lists containing predicted scores. Each sublist corresponds to the relevance scores for items
            matching the order in `label_lists`.
        :param topk : int
            The number of top predictions to consider when calculating the metrics.

        :return: Dict[str, float]
            A dictionary containing the average NDCG, HR, and MRR scores at the given `topk`, with keys formatted as 'NDCG@{topk}',
            'HR@{topk}', and 'MRR@{topk}' respectively. Scores are rounded to 4 decimal places.
        """
        score = {}
        ndcgs = []
        hrs = []
        mrrs = []
        for lab, sco in zip(label_lists, score_lists):
            ndcg = NDCG(lab, lab, topk)
            hr = HR(lab, lab, topk)
            mrr = MRR(lab, topk)
            ndcgs.append(ndcg)
            hrs.append(hr)
            mrrs.append(mrr)

        # compute metrics
        score[f'NDCG@{topk}'] = np.round(np.mean(ndcgs), 4)
        score[f'HR@{topk}'] = np.round(np.mean(hrs), 4)
        score[f'MRR@{topk}'] = np.round(np.mean(mrrs), 4)
        return score

    def get_categories(self, iid2pid):
        return list(set(iid2pid.values()))

    def cal_fair_score(self, iid2pid, predict, topk):
        """
        Calculate fairness scores for a recommendation system's evaluation.

        This method computes various fairness metrics at a specified top-k cutoff to evaluate
        the diversity and inclusiveness of the predicted items. It utilizes different fairness
        measures like MMF (Max-Min Fairness), Gini coefficient, Min-Max Ratio, and Entropy to
        quantify the balance across different categories or groups within the predictions.

        :param iid2pid (Dict[int, int]): A mapping where keys are item IDs and values are their respective group/category IDs.
        :param predict (List[Tuple[int, float]]): A list of tuples, each containing an item ID and its predicted score/score.
        :param topk (int): The top-k count used to consider the highest scored items for fairness evaluation.

        :return:
        A dictionary with keys as the metric names prefixed with the top-k cutoff (e.g., 'MMF@5') and values as the
        corresponding calculated scores, rounded to 4 decimal places.
        """
        #
        score = {}
        cates_value = self.get_cates_value(iid2pid, predict, topk)
        # print(cates_value)
        mmf = MMF(cates_value)
        cate_gini = Gini(cates_value)
        maxmin_ratio = MinMaxRatio(cates_value)
        # cv = (cates_value)
        entropy = Entropy(cates_value)
        score[f'MMF@{topk}'] = np.round(mmf, 4)
        score[f'Gini@{topk}'] = np.round(cate_gini, 4)
        score[f'MMR@{topk}'] = np.round(maxmin_ratio, 4)
        # score[f'cv@{topk}'] = np.round(cv, 4)
        score[f'Entropy@{topk}'] = np.round(entropy, 4)
        return score

    def llm_eval(self, grounding_result, iid2pid):
        """
        Evaluate the performance of a language model based on grounding results and item-pid mappings.

        This method assesses the accuracy and fairness of the model's predictions at different top-K thresholds.
        It computes both accuracy scores and fairness scores,汇总 these into a comprehensive evaluation result.

        :param grounding_result (Dict[str, Any]): The output from the model grounding process, containing necessary information for evaluation.
        :param iid2pid (Dict[str, str]): A mapping from item IDs to product IDs, used in calculating fairness metrics.

        :return:
        - eval_result (Dict[str, float]): A dictionary summarizing the evaluation outcomes, including accuracy and fairness scores for each specified top-K value.
        """
        predict_lists, label_lists, score_lists = self.get_data(grounding_result)
        eval_result = {}
        for topk in self.topk_list:
            acc_score = self.cal_acc_score(label_lists, score_lists, topk)
            fair_score = self.cal_fair_score(iid2pid, predict_lists, topk)
            eval_result.update(acc_score)
            eval_result.update(fair_score)
            # eval_result.update({f'Top{topk}': acc_score})
        print(f'Evaluate_result:{eval_result}')
        return eval_result
