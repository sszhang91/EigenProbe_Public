import logging
import warnings
import Spectral as S
warnings.simplefilter("ignore")

def test_Nq(Nq):
    logging.info("Nq={}".format(Nq))
    N=400;L=500;
    dx=L/N;Rx=0.85*L;Dx=(L-Rx)/2;
    T=S.Timer()
    SL = S.SpectralLaplacian_ribbon(Lx=L,Ly=L,Nx=N,Nqmax=Nq,Rx=Rx,x0=Dx)
    logging.info(T())

class TestSpectralTiming(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(name)-12s %(message)s",
                            datefmt="%m-%d %H:%M",
                            filename="./timing_tests.log",
                            filemode="w")

    def test_Nq_100(self):
        test_Nq(100)

    def test_Nq_500(self):
        test_Nq(500)

    def test_Nq_1000(self):
        test_Nq(1000)

    @unittest.skip("Skipping Nq=5000 test for time...")
    def test_Nq_5000(self):
        test_Nq(5000)
