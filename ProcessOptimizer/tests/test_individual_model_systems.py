import unittest

import numpy as np
from ProcessOptimizer.model_systems import branin
from ProcessOptimizer.space import Real


class TestModelSystem(unittest.TestCase):
    # Make a test for the branin ModelSystem
    def test_branin(self):
        assert branin.noise_size == 0.02
        assert len(branin.space.dimensions) == 2
        assert branin.space.dimensions[0].name == "x1"
        assert branin.space.dimensions[1].name == "x2"
        assert isinstance(branin.space.dimensions[0], Real)
        assert isinstance(branin.space.dimensions[0], Real)
        assert branin.space.dimensions[0].bounds == (-5, 10)
        assert branin.space.dimensions[1].bounds == (0, 15)
        self.assertAlmostEqual(branin.get_score([0, 0]), 55.60820698386535, places=5)
        # This also tests the seeding of the random state
        branin.noise_size = 0
        assert branin.get_score([-np.pi, 12.275]) == 0.39788735772973816
