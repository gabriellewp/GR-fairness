"""
LLM-Reranker: Semantic-Aware Post-Processing for Fair Recommendation

This module implements a novel approach to fairness in recommendation systems by moving from
mathematical fairness (Gini, MMF) to semantic/cognitive fairness using Large Language Models.
"""

import numpy as np
import json
from .Abstract_Reranker import Abstract_Reranker
from ..llm.llm_caller import LLM_caller
from ..metric import Gini, MMF
from tqdm import tqdm
import warnings


class LLM_Reranker(Abstract_Reranker):
    """
    LLM-based reranker that uses Chain-of-Thought reasoning to achieve semantic fairness.
    
    This reranker goes beyond traditional ID-based reranking by:
    1. Understanding the semantic content and meaning of items
    2. Explaining fairness decisions through reasoning chains
    3. Validating fairness claims against mathematical metrics (Gini, MMF)
    """
    
    def __init__(self, config, weights=None, item_metadata=None, use_guardrail=True):
        """
        Initialize the LLM-based semantic reranker.
        
        :param config: Configuration dictionary containing model settings
        :param weights: Optional group weights for fairness computation
        :param item_metadata: Dictionary mapping item IDs to their semantic features
                             (e.g., {item_id: {'title': ..., 'category': ..., 'description': ...}})
        :param use_guardrail: Whether to use the fairness-hallucination guardrail
        """
        super().__init__(config, weights)
        
        # Initialize LLM
        self.llm_caller = LLM_caller(config)
        self.item_metadata = item_metadata or {}
        self.use_guardrail = use_guardrail
        
        # Fairness validation thresholds
        self.gini_threshold = config.get('gini_threshold', 0.4)  # Max acceptable Gini
        self.mmf_threshold = config.get('mmf_threshold', 0.3)    # Min acceptable MMF
        self.max_retries = config.get('max_llm_retries', 3)
        
        # Chain-of-Thought prompt template
        self.cot_prompt_template = config.get('cot_prompt_template', self._default_cot_prompt())
        
        # Logging
        self.rerank_logs = []
        
    def _default_cot_prompt(self):
        """
        Default Chain-of-Thought prompt template for fairness-aware reranking.
        """
        return """You are an expert recommendation system focused on fairness and diversity.

**Task**: Re-rank the following items to maximize both relevance and fairness.

**User Context**: {user_context}

**Current Ranking** (Top {k} items):
{current_ranking}

**Group Distribution**:
{group_distribution}

**Instructions**:
1. **Analyze**: Examine the current ranking for potential biases. Consider:
   - Are certain groups (categories/demographics) over-represented?
   - Is there semantic diversity in perspectives and content?
   - Are minority groups getting fair exposure?

2. **Reason**: Explain step-by-step:
   - What fairness issues do you observe?
   - Which items should be promoted/demoted and why?
   - How does your reranking improve semantic diversity?

3. **Rerank**: Provide a new ranking that balances relevance with fairness.
   Output format: A JSON list of item IDs in the new order.

**Think step-by-step and explain your reasoning:**
"""

    def _construct_item_description(self, item_id):
        """
        Construct a semantic description of an item from metadata.
        
        :param item_id: The item identifier
        :return: String description of the item
        """
        if item_id not in self.item_metadata:
            return f"Item {item_id}"
        
        meta = self.item_metadata[item_id]
        desc_parts = [f"Item {item_id}"]
        
        if 'title' in meta:
            desc_parts.append(f"Title: {meta['title']}")
        if 'category' in meta or 'publisher' in meta:
            cat = meta.get('category', meta.get('publisher', 'Unknown'))
            desc_parts.append(f"Category: {cat}")
        if 'description' in meta:
            desc_parts.append(f"Description: {meta['description'][:200]}...")
            
        return " | ".join(desc_parts)
    
    def _format_ranking_for_llm(self, item_ids, scores=None):
        """
        Format a ranking list for LLM consumption.
        
        :param item_ids: List of item IDs in ranking order
        :param scores: Optional relevance scores
        :return: Formatted string
        """
        ranking_str = ""
        for idx, item_id in enumerate(item_ids):
            item_desc = self._construct_item_description(item_id)
            score_str = f" (score: {scores[idx]:.4f})" if scores is not None else ""
            ranking_str += f"{idx + 1}. {item_desc}{score_str}\n"
        return ranking_str
    
    def _get_group_distribution(self, item_ids):
        """
        Calculate group distribution in the current ranking.
        
        :param item_ids: List of item IDs
        :return: Dictionary with group counts and formatted string
        """
        group_counts = {}
        for item_id in item_ids:
            if item_id in self.iid2pid:
                group_id = self.iid2pid[item_id]
                group_counts[group_id] = group_counts.get(group_id, 0) + 1
        
        total = len(item_ids)
        dist_str = "Group Distribution:\n"
        for group_id, count in sorted(group_counts.items()):
            percentage = (count / total) * 100
            dist_str += f"  Group {group_id}: {count} items ({percentage:.1f}%)\n"
        
        return {"counts": group_counts, "formatted": dist_str}
    
    def _llm_rerank_with_cot(self, user_id, top_k_items, top_k_scores, user_context=""):
        """
        Perform LLM-based reranking using Chain-of-Thought reasoning.
        
        :param user_id: User identifier
        :param top_k_items: List of top-k item IDs
        :param top_k_scores: Relevance scores for top-k items
        :param user_context: Optional user context/history
        :return: Reranked list of item IDs and reasoning
        """
        k = len(top_k_items)
        
        # Format current ranking
        current_ranking_str = self._format_ranking_for_llm(top_k_items, top_k_scores)
        
        # Get group distribution
        group_info = self._get_group_distribution(top_k_items)
        
        # Construct prompt
        prompt = self.cot_prompt_template.format(
            user_context=user_context or f"User {user_id}",
            k=k,
            current_ranking=current_ranking_str,
            group_distribution=group_info["formatted"]
        )
        
        # Call LLM
        try:
            response = self.llm_caller.llm_func(prompt)
            
            # Parse response to extract reasoning and reranked list
            reranked_items, reasoning = self._parse_llm_response(response, top_k_items)
            
            return reranked_items, reasoning, response
            
        except Exception as e:
            warnings.warn(f"LLM reranking failed for user {user_id}: {str(e)}")
            return top_k_items, f"Error: {str(e)}", ""
    
    def _parse_llm_response(self, response, original_items):
        """
        Parse LLM response to extract reranked items and reasoning.
        
        :param response: Raw LLM response
        :param original_items: Original list of items (fallback)
        :return: (reranked_items, reasoning)
        """
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\[[\s\S]*?\]', response)
            
            if json_match:
                reranked_items = json.loads(json_match.group())
                # Validate that all items are in original list
                if all(item in original_items for item in reranked_items):
                    # Extract reasoning (everything before the JSON)
                    reasoning = response[:json_match.start()].strip()
                    return reranked_items, reasoning
            
            # Fallback: return original order
            warnings.warn("Could not parse LLM reranking response, using original order")
            return original_items, response
            
        except Exception as e:
            warnings.warn(f"Error parsing LLM response: {str(e)}")
            return original_items, response
    
    def _validate_fairness(self, reranked_items):
        """
        Fairness-Hallucination Guardrail: Validate fairness using mathematical metrics.
        
        This checks whether the LLM's claimed fairness improvements are real
        by computing Gini Index and MMF on the reranked results.
        
        :param reranked_items: List of reranked item IDs
        :return: (is_fair, metrics_dict)
        """
        # Compute group utilities (exposure)
        group_utilities = np.zeros(self.group_num)
        
        for rank, item_id in enumerate(reranked_items):
            if item_id in self.iid2pid:
                group_id = self.iid2pid[item_id]
                # Exposure decreases with rank (position bias)
                exposure = 1.0 / np.log2(rank + 2)  # +2 to avoid log2(1)=0
                group_utilities[group_id] += exposure
        
        # Calculate fairness metrics
        gini = Gini(group_utilities, weights=self.weights)
        mmf = MMF(group_utilities, ratio=0.5, weights=self.weights)
        
        # Validate against thresholds
        is_fair = (gini <= self.gini_threshold) and (mmf >= self.mmf_threshold)
        
        metrics = {
            'gini': gini,
            'mmf': mmf,
            'is_fair': is_fair,
            'gini_threshold': self.gini_threshold,
            'mmf_threshold': self.mmf_threshold
        }
        
        return is_fair, metrics
    
    def rerank(self, ranking_score, k, user_contexts=None):
        """
        Re-rank items using LLM-based semantic fairness reasoning.
        
        This is the main entry point for the reranker. It:
        1. Gets top-k items from initial ranking
        2. Uses LLM with Chain-of-Thought to rerank
        3. Validates fairness using mathematical guardrails
        4. Retries if fairness validation fails
        
        :param ranking_score: Array of shape (n_users, n_items) with relevance scores
        :param k: Number of items to return per user
        :param user_contexts: Optional list of user context strings
        :return: List of reranked item lists for each user
        """
        n_users = ranking_score.shape[0]
        reranked_results = []
        
        user_contexts = user_contexts or [""] * n_users
        
        for user_idx in tqdm(range(n_users), desc="LLM Reranking"):
            user_scores = ranking_score[user_idx]
            
            # Get top-k items by score
            top_k_indices = np.argsort(user_scores)[::-1][:k]
            top_k_items = top_k_indices.tolist()
            top_k_scores = user_scores[top_k_indices]
            
            # Attempt LLM reranking with retries
            best_reranked = top_k_items
            best_fairness = -1
            best_reasoning = ""
            
            for attempt in range(self.max_retries):
                reranked_items, reasoning, raw_response = self._llm_rerank_with_cot(
                    user_idx, top_k_items, top_k_scores, user_contexts[user_idx]
                )
                
                # Validate fairness if guardrail is enabled
                if self.use_guardrail:
                    is_fair, metrics = self._validate_fairness(reranked_items)
                    
                    # Log the attempt
                    log_entry = {
                        'user_id': user_idx,
                        'attempt': attempt + 1,
                        'original_ranking': top_k_items,
                        'reranked': reranked_items,
                        'reasoning': reasoning,
                        'fairness_metrics': metrics,
                        'passed_guardrail': is_fair
                    }
                    self.rerank_logs.append(log_entry)
                    
                    # Accept if fair or if better than previous attempts
                    if is_fair or metrics['gini'] < best_fairness or best_fairness == -1:
                        best_reranked = reranked_items
                        best_fairness = metrics['gini']
                        best_reasoning = reasoning
                        
                        if is_fair:
                            break  # Success!
                else:
                    # No guardrail, accept first result
                    best_reranked = reranked_items
                    best_reasoning = reasoning
                    break
            
            reranked_results.append(best_reranked)
        
        return reranked_results
    
    def save_logs(self, filepath):
        """
        Save reranking logs for analysis.
        
        :param filepath: Path to save logs (JSON format)
        """
        with open(filepath, 'w') as f:
            json.dump(self.rerank_logs, f, indent=2)
        print(f"Reranking logs saved to {filepath}")
    
    def get_fairness_summary(self):
        """
        Get summary statistics of fairness validation.
        
        :return: Dictionary with summary statistics
        """
        if not self.rerank_logs:
            return {}
        
        total_attempts = len(self.rerank_logs)
        passed = sum(1 for log in self.rerank_logs if log.get('passed_guardrail', False))
        
        gini_scores = [log['fairness_metrics']['gini'] for log in self.rerank_logs 
                      if 'fairness_metrics' in log]
        mmf_scores = [log['fairness_metrics']['mmf'] for log in self.rerank_logs 
                     if 'fairness_metrics' in log]
        
        return {
            'total_rerankings': total_attempts,
            'passed_guardrail': passed,
            'pass_rate': passed / total_attempts if total_attempts > 0 else 0,
            'avg_gini': np.mean(gini_scores) if gini_scores else None,
            'avg_mmf': np.mean(mmf_scores) if mmf_scores else None,
            'min_gini': np.min(gini_scores) if gini_scores else None,
            'max_gini': np.max(gini_scores) if gini_scores else None,
        }
