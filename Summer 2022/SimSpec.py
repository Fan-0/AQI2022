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
    test = qs.loadData(args[0])
    backend = FakeOpenPulse2Q()
    pulse = qs.Custom_Fgp('low_freq',test,backend)
    #data = qs.Cgmp_Spec(pulse,0.2,0.98,10,name=str(args[0]).split('\\')[1][:-2])
    data = []
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    for i in range(int(args[1])):
        data.append(qs.Spec(pulse,i+1,0.001,0.5,10,option=1,name=str(args[0]).split('\\')[1][:-2]))
    for i in range(int(args[1])):
        ax.plot(data[i].all_probs[:,0], data[i].all_probs[:,1],label='N='+str(i))
    ax.legend()
    plt.show()
    name = ("spec_series"+"_"+args[0].split("\\")[1].split(".")[0]+"_"+str(str(datetime.datetime.now())[11:]+".p").replace(":", "") )
    file = open(str(name), 'wb')
    pickle.dump(data,file)
    file.close()
    
  
if __name__=="__main__":
    main()