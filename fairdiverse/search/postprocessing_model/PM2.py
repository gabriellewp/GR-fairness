import os
import json
import pickle
import numpy as np
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
@inproceedings{dang2012diversity,
  title={Diversity by proportionality: an election-based approach to search result diversification},
  author={Dang, Van and Croft, W Bruce},
  booktitle={Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval},
  pages={65--74},
  year={2012}
}
'''

class PM2(BasePostProcessUnsupervisedModel):
    def __init__(self, top_k=20):
        """ 
        Initializes the PM2 for diversified document ranking.

        :param top_k: The maximum number of documents to be ranked. Default is 20.
        """
        super().__init__(top_k)
    
    def rerank(self, config):
        """ 
        Re-ranks documents using the PM-2 algorithm based on BM25 scores and query diversification.

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
            doc_ranking = self.calculate_pm2_score(qid, bm25_dict, config['lambda'])

            if doc_ranking == []:
              doc_ranking = qd[qid].doc_list[:self.top_k]
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
        print(f'\n=== PM2 Results at K={self.top_k} ===')
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



    def calculate_pm2_score(self, qid, bm25_dict, _lambda=0.5):
        """ 
        Implements the PM-2 algorithm for diversified document ranking.

        :param qid: Query ID for which documents are being ranked.
        :param bm25_dict: Dictionary containing BM25 scores for both original and suggestion queries.
        :param _lambda: A trade-off parameter (0 to 1) that balances relevance and diversity. 
                        A higher value favors relevance, while a lower value emphasizes diversity.
        :return: A list of selected document IDs in diversified ranking order.
        """
        # Initialize variables
        doc_list = list(bm25_dict[qid][0].keys())  # Available documents
        suggestion_scores = bm25_dict[qid][1]  # Query suggestions' BM25 scores
        num_suggestions = len(suggestion_scores)
        if num_suggestions == 0:
          return []
        
        # Initialize seat allocation for each aspect (query suggestion)
        s = np.zeros(num_suggestions)
        
        # Initialize selected documents set and remaining documents
        S = []  # Selected documents
        R = doc_list.copy()  # Remaining documents
        
        # Main loop for selecting documents
        while len(S) < self.top_k and R:
            # Calculate quotient for each aspect
            quotient = np.zeros(num_suggestions)
            for i in range(num_suggestions):
                # v_i is assumed to be 1 for all aspects (uniform importance)
                v_i = 1
                quotient[i] = v_i / (2 * s[i] + 1)
                
            # Find aspect with maximum quotient
            i_star = np.argmax(quotient)
            
            # Calculate score for each remaining document
            best_score = float('-inf')
            d_star = None
            
            for d_j in R:
                doc_idx = doc_list.index(d_j)
                
                # Calculate first term (relevance to chosen aspect)
                first_term = _lambda * quotient[i_star] * suggestion_scores[i_star][doc_idx]
                
                # Calculate second term (relevance to other aspects)
                second_term = (1 - _lambda) * sum(
                    quotient[i] * suggestion_scores[i][doc_idx]
                    for i in range(num_suggestions)
                    if i != i_star
                )
                
                # Combined score
                score = first_term + second_term
                
                if score > best_score:
                    best_score = score
                    d_star = d_j
            
            if d_star is None:
                break
                
            # Update sets
            S.append(d_star)
            R.remove(d_star)
            
            # Update seat allocations
            doc_idx = doc_list.index(d_star)
            total_relevance = sum(suggestion_scores[j][doc_idx] for j in range(num_suggestions))
            
            for i in range(num_suggestions):
                if total_relevance > 0:  # Avoid division by zero
                    s[i] += suggestion_scores[i][doc_idx] / total_relevance
        
        return S