# Copyright 2026 The NEST Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
import numpy as np
from pyscf import lib

from nest.soc import SOCBase, SFTDASOC, SpinFreeState
from nest.soc.sftda import SOC as SFTDASOCImpl
from nest.soc.soc import clebsch_gordan_rank1


class KnownValues(unittest.TestCase):
    def test_soc_block_format(self):
        bra = SpinFreeState(None, None, 0.0, 0.0, None, 'bra')
        ket = SpinFreeState(None, None, 0.0, 0.5, None, 'ket')
        lines = SOCBase._format_soc_block(
            np.array([[1.25 - 2.5j, -3.75 + 4.5j]]), bra, ket,
        )

        self.assertEqual(lines[0], 'SOC matrix elements (cm^-1):')
        self.assertIn('ket M_S= -0.5', lines[1])
        self.assertIn('ket M_S=  0.5', lines[1])
        self.assertIn('bra M_S=  0.0', lines[3])
        self.assertIn('(  1.250000, -2.500000)', lines[3])
        self.assertIn('( -3.750000,  4.500000)', lines[3])

    def test_soc_class_names(self):
        self.assertTrue(issubclass(SOCBase, lib.StreamObject))
        self.assertIs(SFTDASOC, SFTDASOCImpl)

    def test_rank_one_clebsch_gordan(self):
        self.assertAlmostEqual(clebsch_gordan_rank1(0, 0, 1, 1, 1), 1.0)
        self.assertAlmostEqual(clebsch_gordan_rank1(1, 1, 0, 1, 1), 2 ** -0.5)
        self.assertAlmostEqual(clebsch_gordan_rank1(1, 0, 1, 1, 1), -2 ** -0.5)
        self.assertAlmostEqual(clebsch_gordan_rank1(1, 0, 0, 0, 0), -3 ** -0.5)


if __name__ == '__main__':
    unittest.main()
