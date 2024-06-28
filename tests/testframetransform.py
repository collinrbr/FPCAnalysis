import unittest

import FPCAnalysis
import numpy as np

class TestFrameTransform(unittest.TestCase):
    
    #boost into shock rest frame, where ez (i.e. direction that is not motional electric field or cross shock potential)
    #  should be approximately constant
    def test_fields_transform(self):
        passed = True
        dfields = FPCAnalysis.ddhr.field_loader(path='tests/testdata/dHybridR/M06_th45/',num='2000')

        vx = 1.881940772078266
        dfields = FPCAnalysis.ft.lorentz_transform_vx(dfields,vx)
        dfavg = FPCAnalysis.anl.get_average_fields_over_yz(dfields)

        mean_dfavgtot = np.mean(np.sqrt(dfavg['ez'][0,0,:]**2+dfavg['ey'][0,0,:]**2+dfavg['ex'][0,0,:]**2))

        tol = .15
        if(np.abs(np.std(dfavg['ez'][0,0,:])) < tol*mean_dfavgtot):
            passed = True
        else:
            passed = False

        self.assertEqual(passed, True)


if __name__ == '__main__':
    unittest.main()
