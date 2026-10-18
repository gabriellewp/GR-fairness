"""
LLM-Reranker for Search: Semantic-Aware Post-Processing

This module implements semantic fairness for search/information retrieval by using
Large Language Models to understand document content and ensure diverse perspectives.

Unlike traditional post-processing methods (xQuAD, PM2, etc.) that treat documents as IDs,
this approach uses LLMs to:
1. Understand the semantic content of documents
2. Reason about fairness and diversity in search results
3. Validate claims using mathematical fairness metrics

"""

import os
import json
import pickle
import numpy as np
from .base import BasePostProcessUnsupervisedModel
from ..utils.utils import pkl_load
from tqdm import tqdm
import warnings


class LLM_Reranker_Search(BasePostProcessUnsupervisedModel):
    """
    LLM-based reranker for search that uses Chain-of-Thought reasoning for semantic diversity.
    
    This reranker analyzes document content semantically to ensure:
    - Multiple perspectives are represented
    - No single viewpoint/group dominates
    - Fairness is validated with Gini and MMF metrics
    """
    
    def __init__(self, top_k=20, llm_config=None, use_guardrail=True):
        """
        Initialize the LLM-based search reranker.
        
        :param top_k: Number of documents to return
        :param llm_config: Configuration for LLM (model name, API key, etc.)
        :param use_guardrail: Whether to validate fairness with mathematical metrics
        """
        super().__init__(top_k)
        
        # Initialize LLM
        if llm_config is None:
            raise ValueError("llm_config is required for LLM_Reranker_Search")
        
        self.llm_config = llm_config
        self.use_guardrail = use_guardrail
        
        # Import here to avoid circular dependency
        from ...recommendation.llm.llm_caller import LLM_caller
        self.llm_caller = LLM_caller(llm_config)
        
        # Fairness thresholds
        self.gini_threshold = llm_config.get('gini_threshold', 0.4)
        self.mmf_threshold = llm_config.get('mmf_threshold', 0.3)
        self.max_retries = llm_config.get('max_llm_retries', 2)
        
        # Prompt template
        self.cot_prompt_template = llm_config.get('cot_prompt_template', 
                                                   self._default_search_cot_prompt())
        
        # Logging
        self.rerank_logs = []
        self.document_cache = {}
        
    def _default_search_cot_prompt(self):
        """
        Default Chain-of-Thought prompt for search result diversification.
        """
        return """You are an expert search engine focused on fairness and diversity in information retrieval.

**User Query**: {query}

**Current Top {k} Results**:
{current_ranking}

**Group/Category Distribution**:
{group_distribution}

**Task**: Re-rank these search results to provide:
1. **Relevance**: Results should still be relevant to the user's query
2. **Diversity**: Results should cover multiple perspectives, viewpoints, and subtopics
3. **Fairness**: No single group/category should dominate; minority perspectives deserve representation

**Instructions**:
1. **Analyze**: 
   - What topics/perspectives are currently over-represented?
   - Which viewpoints are missing or under-represented?
   - Are there semantic redundancies (similar content)?

2. **Reason** (think step-by-step):
   - Why is the current ranking potentially unfair or not diverse?
   - Which documents provide unique perspectives?
   - How can we balance relevance with diversity?

3. **Rerank**: 
   - Provide a new ranking as a JSON list of document IDs
   - Balance relevance with semantic diversity
   - Ensure fair group representation

**Output Format**: 
First explain your reasoning, then provide the reranked list as JSON: ["doc_id_1", "doc_id_2", ...]

**Your response**:
"""

    def _load_document_features(self, config, doc_id):
        """
        Load document features (title, snippet, category, etc.).
        
        :param config: Configuration dictionary
        :param doc_id: Document identifier
        :return: Dictionary with document features
        """
        # Check cache first
        if doc_id in self.document_cache:
            return self.document_cache[doc_id]
        
        # Load from data directory (implementation depends on your data format)
        # This is a placeholder - adapt to your data structure
        data_dir = config.get('data_dir', '')
        
        try:
            # Try to load from metadata file
            metadata_path = os.path.join(data_dir, 'doc_metadata.json')
            if os.path.exists(metadata_path) and not self.document_cache:
                with open(metadata_path, 'r') as f:
                    all_metadata = json.load(f)
                    self.document_cache = all_metadata
                    return all_metadata.get(doc_id, {'id': doc_id, 'title': f'Document {doc_id}'})
            
            # Fallback: minimal info
            return {'id': doc_id, 'title': f'Document {doc_id}'}
            
        except Exception as e:
            warnings.warn(f"Could not load metadata for {doc_id}: {str(e)}")
            return {'id': doc_id, 'title': f'Document {doc_id}'}
    
    def _format_documents_for_llm(self, doc_ids, scores, config):
        """
        Format document list for LLM with semantic information.
        
        :param doc_ids: List of document IDs
        :param scores: Relevance scores
        :param config: Configuration
        :return: Formatted string
        """
        doc_str = ""
        for idx, (doc_id, score) in enumerate(zip(doc_ids, scores)):
            doc_info = self._load_document_features(config, doc_id)
            
            title = doc_info.get('title', 'Unknown')
            snippet = doc_info.get('snippet', doc_info.get('abstract', ''))[:200]
            category = doc_info.get('category', doc_info.get('group', 'Unknown'))
            
            doc_str += f"\n{idx + 1}. [{doc_id}] (Score: {score:.4f})\n"
            doc_str += f"   Title: {title}\n"
            if snippet:
                doc_str += f"   Snippet: {snippet}...\n"
            doc_str += f"   Category: {category}\n"
        
        return doc_str
    
    def _get_group_distribution_search(self, doc_ids, config):
        """
        Calculate group/category distribution in search results.
        
        :param doc_ids: List of document IDs
        :param config: Configuration
        :return: Formatted distribution string and counts
        """
        group_counts = {}
        
        for doc_id in doc_ids:
            doc_info = self._load_document_features(config, doc_id)
            category = doc_info.get('category', doc_info.get('group', 'Unknown'))
            group_counts[category] = group_counts.get(category, 0) + 1
        
        total = len(doc_ids)
        dist_str = ""
        for category, count in sorted(group_counts.items()):
            percentage = (count / total) * 100
            dist_str += f"  {category}: {count} documents ({percentage:.1f}%)\n"
        
        return dist_str, group_counts
    
    def _calculate_gini_coefficient(self, group_counts):
        """
        Calculate Gini coefficient for group distribution.
        
        :param group_counts: Dictionary of group counts
        :return: Gini coefficient
        """
        if not group_counts:
            return 0.0
        
        values = np.array(list(group_counts.values()))
        values = np.sort(values)
        n = len(values)
        
        cumulative_sum = np.cumsum(values)
        gini = (n + 1 - 2 * (np.sum(cumulative_sum) / cumulative_sum[-1])) / n
        
        return gini
    
    def _calculate_mmf(self, group_counts, ratio=0.5):
        """
        Calculate Max-Min Fairness for group distribution.
        
        :param group_counts: Dictionary of group counts
        :param ratio: Ratio of minimum groups to consider
        :return: MMF score
        """
        if not group_counts:
            return 0.0
        
        values = np.array(list(group_counts.values()))
        values = np.sort(values)
        
        mmf_length = max(1, int(ratio * len(values)))
        mmf = np.sum(values[:mmf_length]) / np.sum(values)
        
        return mmf
    
    def _validate_fairness_search(self, doc_ids, config):
        """
        Fairness-Hallucination Guardrail for search results.
        
        :param doc_ids: Reranked document IDs
        :param config: Configuration
        :return: (is_fair, metrics_dict)
        """
        _, group_counts = self._get_group_distribution_search(doc_ids, config)
        
        gini = self._calculate_gini_coefficient(group_counts)
        mmf = self._calculate_mmf(group_counts)
        
        is_fair = (gini <= self.gini_threshold) and (mmf >= self.mmf_threshold)
        
        return is_fair, {
            'gini': gini,
            'mmf': mmf,
            'is_fair': is_fair,
            'group_counts': group_counts
        }
    
    def _parse_llm_response(self, response, original_docs):
        """
        Parse LLM response to extract reasoning and reranked documents.
        
        :param response: Raw LLM output
        :param original_docs: Original document list (fallback)
        :return: (reranked_docs, reasoning)
        """
        try:
            import re
            
            # Look for JSON list in response
            json_match = re.search(r'\[[\s\S]*?\]', response)
            
            if json_match:
                reranked_docs = json.loads(json_match.group())
                
                # Validate: all docs should be in original list
                if all(doc in original_docs for doc in reranked_docs):
                    reasoning = response[:json_match.start()].strip()
                    return reranked_docs, reasoning
            
            warnings.warn("Could not parse LLM response, using original ranking")
            return original_docs, response
            
        except Exception as e:
            warnings.warn(f"Error parsing LLM response: {str(e)}")
            return original_docs, response
    
    def _llm_rerank_query(self, query, doc_ids, scores, config):
        """
        Rerank documents for a single query using LLM.
        
        :param query: Search query string
        :param doc_ids: List of document IDs
        :param scores: Relevance scores
        :param config: Configuration
        :return: (reranked_docs, reasoning, raw_response)
        """
        k = len(doc_ids)
        
        # Format documents for LLM
        doc_str = self._format_documents_for_llm(doc_ids, scores, config)
        
        # Get group distribution
        dist_str, _ = self._get_group_distribution_search(doc_ids, config)
        
        # Construct prompt
        prompt = self.cot_prompt_template.format(
            query=query,
            k=k,
            current_ranking=doc_str,
            group_distribution=dist_str
        )
        
        # Call LLM
        try:
            response = self.llm_caller.llm_func(prompt)
            reranked_docs, reasoning = self._parse_llm_response(response, doc_ids)
            return reranked_docs, reasoning, response
        except Exception as e:
            warnings.warn(f"LLM call failed for query '{query}': {str(e)}")
            return doc_ids, f"Error: {str(e)}", ""
    
    def rerank(self, config):
        """
        Re-rank search results using LLM-based semantic fairness.
        
        :param config: Configuration dictionary containing:
            - data_dir: Directory with data files
            - model: Model name for output directory
            - queries: Query data
            - initial_scores: Initial ranking scores (e.g., BM25)
        """
        # Load queries and initial rankings
        data_dir = config['data_dir']
        
        # Load initial scores (e.g., BM25)
        score_file = os.path.join(data_dir, config['model'], config.get('score_file', 'bm25_scores.pkl'))
        initial_scores = pkl_load(score_file)
        
        # Load queries
        if 'queries' in config:
            queries = config['queries']
        else:
            query_file = os.path.join(data_dir, 'queries.json')
            with open(query_file, 'r') as f:
                queries = json.load(f)
        
        # Output directory
        output_dir = os.path.join(config.get('tmp_dir', './output'), config['model'], 'llm_reranker')
        os.makedirs(output_dir, exist_ok=True)
        
        # Rerank each query
        for qid, query_text in tqdm(queries.items(), desc="LLM Reranking Queries"):
            # Get initial ranking for this query
            if qid not in initial_scores:
                warnings.warn(f"Query {qid} not in initial scores, skipping")
                continue
            
            doc_scores = initial_scores[qid]
            
            # Sort by score
            sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_docs = [doc_id for doc_id, _ in sorted_items[:self.top_k]]
            top_scores = [score for _, score in sorted_items[:self.top_k]]
            
            # LLM reranking with retries
            best_reranked = top_docs
            best_fairness_score = float('inf')  # Lower Gini is better
            best_reasoning = ""
            
            for attempt in range(self.max_retries):
                reranked_docs, reasoning, raw_response = self._llm_rerank_query(
                    query_text, top_docs, top_scores, config
                )
                
                # Validate fairness
                if self.use_guardrail:
                    is_fair, metrics = self._validate_fairness_search(reranked_docs, config)
                    
                    log_entry = {
                        'query_id': qid,
                        'query_text': query_text,
                        'attempt': attempt + 1,
                        'original_ranking': top_docs,
                        'reranked': reranked_docs,
                        'reasoning': reasoning,
                        'fairness_metrics': metrics,
                        'passed_guardrail': is_fair
                    }
                    self.rerank_logs.append(log_entry)
                    
                    # Keep best result
                    if is_fair or metrics['gini'] < best_fairness_score:
                        best_reranked = reranked_docs
                        best_fairness_score = metrics['gini']
                        best_reasoning = reasoning
                        
                        if is_fair:
                            break
                else:
                    best_reranked = reranked_docs
                    break
            
            # Write output file for this query
            output_file = os.path.join(output_dir, f'{qid}.txt')
            with open(output_file, 'w') as f:
                for rank, doc_id in enumerate(best_reranked):
                    score = len(best_reranked) - rank
                    f.write(f'{qid} Q0 {doc_id} {rank + 1} {score} llm_reranker\n')
        
        # Aggregate results and evaluate if evaluation script is available
        output_txt = os.path.join(output_dir, 'output_best.txt')
        os.system(f'cat {output_dir}/*.txt > {output_txt}')
        
        print(f"LLM reranking complete. Results saved to {output_dir}")
        
        # Save logs
        log_file = os.path.join(output_dir, 'reranking_logs.json')
        self.save_logs(log_file)
        
        return output_txt
    
    def save_logs(self, filepath):
        """
        Save reranking logs for analysis.
        
        :param filepath: Path to save JSON logs
        """
        with open(filepath, 'w') as f:
            json.dump(self.rerank_logs, f, indent=2)
        print(f"Reranking logs saved to {filepath}")
    
    def get_fairness_summary(self):
        """
        Get summary statistics of fairness validation.
        
        :return: Dictionary with summary stats
        """
        if not self.rerank_logs:
            return {}
        
        total = len(self.rerank_logs)
        passed = sum(1 for log in self.rerank_logs if log.get('passed_guardrail', False))
        
        gini_scores = [log['fairness_metrics']['gini'] for log in self.rerank_logs]
        mmf_scores = [log['fairness_metrics']['mmf'] for log in self.rerank_logs]
        
        return {
            'total_queries': total,
            'passed_guardrail': passed,
            'pass_rate': passed / total if total > 0 else 0,
            'avg_gini': np.mean(gini_scores),
            'avg_mmf': np.mean(mmf_scores),
            'min_gini': np.min(gini_scores),
            'max_gini': np.max(gini_scores)
        }
