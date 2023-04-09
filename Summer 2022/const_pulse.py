import Su22 as qs
import sys
from joblib import Parallel, delayed
import datetime
import pickle
import matplotlib.pyplot as plt
from qiskit.providers.fake_provider import FakeOpenPulse2Q
import os


def main():
    args = sys.argv[1:]
    #pulse = qs.Custom_Fgp("spec",qs.loadData(args[0]))
    test = [float(args[0]) for i in range(int(args[1]))]
    backend = FakeOpenPulse2Q()
    #data = qs.Cgmp_Spec(pulse,0.2,0.98,10,name=str(args[0]).split('\\')[1][:-2])
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    name = str("spec_series"+"_" +args[0]+"_"+ args[1]+"L")
    data = (qs.Spec(test,1,0.15,0.25,100,option=1,name=name))
    ax.plot(data.all_probs[:,0], data.all_probs[:,1])
    plt.show()
    data.dump()
    
  
if __name__=="__main__":
    main()