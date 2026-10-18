import os
import json
import pickle
import shutil
import sys

from ..utils.utils import pkl_load, get_metrics_20, get_metrics_at_k
from .base import BasePostProcessUnsupervisedModel

# Import the div_query class so it's available for unpickling
try:
    from ..utils.div_type import div_query, subtopic
except ImportError:
    from fairdiverse.search.utils.div_type import div_query, subtopic

# Custom unpickler to handle module path changes
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect old module paths to new ones
        if module == 'fairdiverse.search.utils.div_type':
            module = 'fairdiverse.search.utils.div_type'
            # Import from the current package structure
            from ..utils import div_type
            return getattr(div_type, name)
        return super().find_class(module, name)

'''
@inproceedings{santos2010exploiting,
  title={Exploiting query reformulations for web search result diversification},
  author={Santos, Rodrygo LT and Macdonald, Craig and Ounis, Iadh},
  booktitle={Proceedings of the 19th international conference on World wide web},
  pages={881--890},
  year={2010}
}
'''

class xQuAD(BasePostProcessUnsupervisedModel):
    def __init__(self, top_k=20):
        """ 
        Initializes the xQuAD for diversified document ranking.

        :param top_k: The maximum number of documents to be ranked. Default is 20.
        """
        super().__init__(top_k)
    
    def rerank(self, config):
        """ 
        Re-ranks documents using the xQuAD algorithm based on BM25 scores and query diversification.

        :param config: A dictionary containing configuration parameters.
        """
        # Get top_k from config, or use default from initialization
        if 'top_k' in config:
            self.top_k = config['top_k']
        
        # Load pickle file with custom unpickler to handle missing modules
        try:
            with open(os.path.join(config['data_dir'], 'div_query.data'), 'rb') as f:
                qd = CustomUnpickler(f).load()
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            raise
            
        bm25_dict = pkl_load(os.path.join(config['data_dir'], config['model'], 'bm25_scores.pkl'))

        # Make output directory unique per K value to avoid conflicts
        output_dir = os.path.join(config['tmp_dir'], f"{config['model']}_k{self.top_k}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_txt = os.path.join(output_dir, 'output_best.txt')
        csv_path = os.path.join(output_dir, 'result.csv')

        for qid in qd:
            doc_ranking = self.calculate_xquad_score(qid, bm25_dict, config['lambda'])

            judge_f = open(os.path.join(output_dir, str(qid)+'.txt'), 'w')
            for i in range(len(doc_ranking)):
                doc_ranking_score = len(doc_ranking) - i
                judge_f.write(str(qid) + ' Q0 ' + doc_ranking[i] + ' ' + str(i + 1) + ' ' + str(doc_ranking_score) + ' indri\n')
            judge_f.close()

        # Concatenate only the numbered query files, not output_best.txt or result.csv
        command = f'cat {output_dir}/[0-9]*.txt > {output_txt}'
        os.system(command)
        command = 'fairdiverse/search/eval/clueweb09/ndeval fairdiverse/search/eval/clueweb09/2009-2012.diversity.ndeval.qrels '+output_txt+' >' + str(csv_path)
        os.system(command)

        # Calculate and print metrics for K=5, 10, 20
        print(f'\n=== xQuAD Results at K={self.top_k} ===')
        alpha_nDCG_k, ERR_IA_k, S_rec_k = get_metrics_at_k(csv_path, self.top_k)
        print(f'alpha-nDCG@{self.top_k} = {alpha_nDCG_k:.4f}, ERR-IA@{self.top_k} = {ERR_IA_k:.4f}, S-rec@{self.top_k} = {S_rec_k:.4f}')
        
        # Also print traditional @20 metrics if K=20
        if self.top_k == 20:
            alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20 = get_metrics_20(csv_path)
            print(f'alpha_nDCG@20_std = {alpha_nDCG_20}, NRBP_20 = {NRBP_20}, ERR_IA_20 = {ERR_IA_20}, S_rec_20 = {S_rec_20}')
        
        # Clean up temporary directory after processing
        print(f'\nCleaning up temporary directory: {output_dir}')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f'Temporary directory cleaned successfully.')



    def calculate_xquad_score(self, qid, bm25_dict, _lambda=0.5):
        """ 
        Implements the xQuAD algorithm for diversified document ranking.

        :param qid: Query ID for which documents are being ranked.
        :param bm25_dict: Dictionary containing BM25 scores for both original and suggestion queries.
        :param _lambda: A trade-off parameter (0 to 1) that balances relevance and diversity. 
                        A higher value favors relevance, while a lower value emphasizes diversity.
        :return: A list of selected document IDs in diversified ranking order.
        """
        original_scores = bm25_dict[qid][0]  # Original query scores
        suggestion_scores = bm25_dict[qid][1]  # Suggestion query scores (n_suggestions x n_docs)
        
        selected_docs = []
        available_docs = set(original_scores.keys())
        
        # Calculate suggestion probabilities (assuming normalized BM25 scores)
        total_suggestion_score = suggestion_scores.sum(axis=0)
        suggestion_probs = suggestion_scores / (total_suggestion_score + 1e-10)  # Add epsilon to avoid division by zero
        
        while len(selected_docs) < self.top_k and available_docs:
            best_score = float('-inf')
            best_doc = None
            
            for doc_id in available_docs:
                # Get original query score component
                original_component = (1 - _lambda) * original_scores[doc_id]
                
                # Calculate expanded component
                expanded_component = 0
                doc_idx = list(original_scores.keys()).index(doc_id)
                
                for qi_idx in range(len(suggestion_scores)):
                    qi_score = suggestion_probs[qi_idx][doc_idx]
                    
                    # Calculate product term for already selected documents
                    product_term = 1
                    for selected_doc in selected_docs:
                        selected_idx = list(original_scores.keys()).index(selected_doc)
                        product_term *= (1 - suggestion_scores[qi_idx][selected_idx])
                    
                    expanded_component += qi_score * product_term
                
                expanded_component *= _lambda
                
                # Combined score
                total_score = original_component + expanded_component
                
                if total_score > best_score:
                    best_score = total_score
                    best_doc = doc_id
            
            if best_doc is None:
                break
                
            selected_docs.append(best_doc)
            available_docs.remove(best_doc)
        
        return selected_docs

