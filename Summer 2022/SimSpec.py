import Su22 as qs
import sys
from joblib import Parallel, delayed
from qiskit.providers.fake_provider import FakeOpenPulse2Q
import os


def main():
    args = sys.argv[1:]
    test = qs.loadData(args[0])
    backend = FakeOpenPulse2Q()
    pulse = qs.Custom_Fgp('low_freq',test,backend)
    data = qs.Spec(pulse,0,1,1000,option=1,name=str(args[0]).split('\\')[1][:-2])
    print(type(data))
    data.draw()
    data.dump()
    
  
if __name__=="__main__":
    main()