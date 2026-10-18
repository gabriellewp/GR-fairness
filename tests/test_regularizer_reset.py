from fairdiverse.recommendation.rank_model.FOCF import FOCF
from fairdiverse.recommendation.rank_model.Reg import Reg


def test_stateless_regularizers_support_trainer_reset_hook():
    # Construction requires dataset metadata, which is irrelevant to this
    # interface test. Neither regularizer has epoch-local state to initialize.
    FOCF.__new__(FOCF).reset_parameters()
    Reg.__new__(Reg).reset_parameters()
