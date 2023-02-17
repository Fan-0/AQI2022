import Su22 as qs
import sys
from joblib import Parallel, delayed
from qiskit.providers.fake_provider import FakeArmonkV2
import os


def main():
    args = sys.argv[1:]
    pulse = qs.loadData(args[0])
    backend = FakeArmonkV2
    data= qs.Spec(pulse,0.02,0.98,100,option=1)
    print(?data)
    
  
if __name__=="__main__":
    main()