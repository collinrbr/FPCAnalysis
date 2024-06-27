import unittest

import FPCAnalysis


class TestLoaddHybridR(unittest.TestCase):
    
    def test_load_fields(self):
        passed = True

        dfields = FPCAnalysis.ddhr.field_loader(path='tests/testdata/dHybridR/M06_th45/',num='2000')

        fieldkeys = ['ex','ey','ez','bx','by','bz']

        #Test that all data keys were loaded
        for fk in fieldkeys:
            if not(fk in dfields.keys()):
                passed = False

        #Test that all coord keys were loaded
        coordkeys = ['_xx','_yy','_zz']
        for fk in fieldkeys:
            for ck in coordkeys:
                if not(fk+ck in dfields.keys()):
                    passed = False

        #Test that size is same for all data keys
        for fk in fieldkeys:
            if (dfields['ex'].shape != dfields[fk].shape):
                passed = False

        #Test that coord size is same for all data keys
        for fk in fieldkeys:
            for ck in coordkeys:
                if not(fk+ck in dfields.keys()):
                    passed = False

        #Check that shape of coords matches shape of fields
        for fk in fieldkeys:
            if not(dfields['ex'].shape == (len(dfields[fk+'_zz']),len(dfields[fk+'_yy']),len(dfields[fk+'_xx']))):
                passed = False

        self.assertEqual(passed, True)


if __name__ == '__main__':
    unittest.main()
