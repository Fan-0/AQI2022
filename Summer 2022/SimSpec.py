import Su22 as qs
import sys
from joblib import Parallel, delayed
from qiskit.providers.fake_provider import FakeArmonkV2
import os


def main():
    args = sys.argv[1:]
    pulse = qs.loadData(args[0])
    data = qs.Spec(pulse,0,1,1000,option=1)
    print(type(data))
    data.draw()
    data.dump()
    
  
if __name__=="__main__":
    main()