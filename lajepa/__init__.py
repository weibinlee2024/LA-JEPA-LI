"""LA-JEPA-LI: compact world-action policies in a frozen V-JEPA 2.1 feature space.

Independent reproduction of La-JEPA (arXiv:2611.08156). Training predicts
future V-JEPA features and an action chunk with two DiTs. Inference drops the
future tokens and integrates only the action flow.
"""

__version__ = "0.1.0"
