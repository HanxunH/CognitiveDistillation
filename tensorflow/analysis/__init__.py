from .cognitive_distillation import CognitiveDistillationAnalysis, min_max_normalization
from .activation_clustering import ACAnalysis
from .abl import ABLAnalysis
from .spectral_signatures import SSAnalysis
from .frequency import FrequencyAnalysis

# LIDAnalysis is not implemented in the original codebase
class LIDAnalysis:
    def analysis(self, features):
        raise NotImplementedError("LIDAnalysis is not implemented")

__all__ = ['CognitiveDistillationAnalysis', 'min_max_normalization', 'ACAnalysis', 'ABLAnalysis', 'SSAnalysis', 'FrequencyAnalysis', 'LIDAnalysis']