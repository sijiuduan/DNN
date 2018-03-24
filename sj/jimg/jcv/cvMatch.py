from jnn.ea import Ea
import numpy as np
import time

def demo():
    A = Ea.load("./55_kpt.pk")
    B = Ea.load("./104_kpt.pk")
    m = []

    print(time.time())
    for a in A.values():
        for b in B.values():
            # c = b.dsp - a.dsp
            c = (b.dsp).astype(np.int16) - (a.dsp).astype(np.int16)
            c = np.abs(c)/256
            # c = np.sum(c)
            pass

    print(time.time())


            # print(c)
            # print(np.sum(c)/32)
            # break
        # break
    print(len(m))
    print(len(A.keys()))
    print(len(B.keys()))


    # Ea.show(B)
    # print((A[31312].dsp).dtype)


    # c  = A[31312].dsp - B[31312].dsp
    # c = np.abs(c.astype(np.int8))
    # print(c)


def demo2():
    A = Ea.load("./55_kpt.pk")
    B = Ea.load("./104_kpt.pk")
    for a in A.values():
        for b in B.values():
            # c = b.dsp - a.dsp
            c = (b.dsp).astype(np.int16) - (a.dsp).astype(np.int16)
            c = np.abs(c)/256
            # c = np.sum(c)
            pass
    pass

if __name__ == '__main__':
    demo2()