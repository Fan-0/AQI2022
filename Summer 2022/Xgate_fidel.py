import Su22 as qs
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qiskit as qk
from qiskit.providers.fake_provider import FakeOpenPulse2Q
import os


def main():
    args = sys.argv[1:]
    #pulse = qs.Custom_Fgp("spec",qs.loadData(args[0]))
    test = qs.loadData(args[0])
    backend = FakeOpenPulse2Q()
    pulse = qs.Custom_Fgp('low_freq',test,backend)
    return qs.X_fidel(pulse,backend, int(args[1]))
  
if __name__=="__main__":
    main()