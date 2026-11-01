"""Check the example's complete difference-index coefficients independently.

Run: python -m unittest discover -s QA -p test_moharam_analytic_convolution.py
The example plots a full spectrum at module scope, so load only its actual
function definition rather than executing that unrelated plotting workload.
"""
import ast
from functools import lru_cache
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz


def example_function():
    path = Path(__file__).resolve().parents[1]/'examples/rcwa/moharam_1D_TE.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                    and n.name=='E_conv_1D_analytic')
    namespace = {'np':np,'toeplitz':toeplitz}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['E_conv_1D_analytic']


def integrated_matrix(order, pattern, period):
    ridge,groove,fill = pattern
    # Preserve the example's existing positive Fourier phase convention.
    center = (1/period)%1
    edges = sorted({0.,1.,(center-fill/2)%1,(center+fill/2)%1})
    def epsilon(x):
        distance = (x-center+.5)%1-.5
        return ridge**2 if abs(distance)<fill/2 else groove**2
    @lru_cache(None)
    def coefficient(difference):
        real=quad(lambda x:epsilon(x)*np.cos(2*np.pi*difference*x),0,1,
                  points=edges[1:-1],epsabs=1e-12,epsrel=1e-12)[0]
        imag=quad(lambda x:epsilon(x)*np.sin(2*np.pi*difference*x),0,1,
                  points=edges[1:-1],epsabs=1e-12,epsrel=1e-12)[0]
        return real+1j*imag
    modes=np.arange(-order,order+1)
    return np.array([[coefficient(int(r-c)) for c in modes] for r in modes])


class AnalyticConvolutionTests(unittest.TestCase):
    def test_all_mode_differences_match_piecewise_material_integrals(self):
        function=example_function()
        patterns=[[3.48,1.,.3],[1.5,2.2,.67]]
        for order in (1,3,5,10):
            for period in (.7,1.3):
                for pattern,actual in zip(patterns,function(order,patterns,period)):
                    with self.subTest(order=order,period=period,pattern=pattern):
                        expected=integrated_matrix(order,pattern,period)
                        self.assertEqual(actual.shape,(2*order+1,2*order+1))
                        np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=2e-11)
                        np.testing.assert_allclose(actual,actual.conj().T,atol=2e-13)
                        self.assertGreaterEqual(np.linalg.eigvalsh(actual).min(),min(pattern[:2])**2-1e-11)
                        self.assertLessEqual(np.linalg.eigvalsh(actual).max(),max(pattern[:2])**2+1e-11)

    def test_uniform_limits_and_zero_order(self):
        function=example_function()
        for order in (0,1,10):
            for pattern,index in [([3.48,1.,0.],1.),([3.48,1.,1.],3.48),([2.,2.,.3],2.)]:
                actual=function(order,[pattern],.7)[0]
                np.testing.assert_allclose(actual,index**2*np.eye(2*order+1),atol=2e-13)
            self.assertAlmostEqual(function(0,[[3.48,1.,.3]],.7)[0][0,0].real,
                                   3.48**2*.3+.7)


if __name__=='__main__':unittest.main()
