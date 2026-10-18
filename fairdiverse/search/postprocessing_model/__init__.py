from .xQuAD import xQuAD
from .DESA import DESA
from .PM2 import PM2
from .DALETOR import DALETOR
from .llm_reranker import LLM_Reranker_Search
from .base import BasePostProcessModel, BasePostProcessUnsupervisedModel

diversity_method_mapping = {
    'DESA': DESA,
    'DALETOR': DALETOR,
    'PM2': PM2,
    'xQuAD': xQuAD,
    'LLM_Reranker': LLM_Reranker_Search
}