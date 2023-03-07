import Su22 as qs
import sys
from joblib import Parallel, delayed
from qiskit.providers.fake_provider import FakeOpenPulse2Q
import os


def main():
    args = sys.argv[1:]
    #pulse = qs.Custom_Fgp("spec",qs.loadData(args[0]))
    test = qs.loadData(args[0])
    backend = FakeOpenPulse2Q()
    pulse = qs.Custom_Fgp('low_freq',test,backend)
    #data = qs.Cgmp_Spec(pulse,0.2,0.98,10,name=str(args[0]).split('\\')[1][:-2])
    data = qs.Spec(pulse,0.001,0.5,100,option=1,name=str(args[0]).split('\\')[1][:-2])
    print(type(data))
    data.draw()
    data.dump()
    
  
if __name__=="__main__":
    main()