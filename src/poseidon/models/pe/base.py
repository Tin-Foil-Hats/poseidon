from __future__ import annotations

from typing import Dict

import torch.nn as nn


class PEBase(nn.Module):
    def feat_dim(self) -> int:
        raise NotImplementedError

    def feature_layout(self) -> Dict[str, int]:
        """Return a mapping describing feature group sizes.

        By default, encoders expose a single "full" group equal to the
        flattened feature dimension. Encoders that emit structured blocks
        (e.g., spatial vs. temporal features) can override this to provide
        richer metadata consumed by downstream nets.
        """

        return {"full": self.feat_dim()}
