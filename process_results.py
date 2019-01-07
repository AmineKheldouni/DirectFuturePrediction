import pandas as pd
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot(exp,variable):
    results = pd.read_csv('./experiments/' + exp + '/logs/' + 'results.csv',
                          sep=',',
                          header=0)

    results.plot(x='Time', y=variable)
    plt.title(variable + 'through episodes')
    plt.show()

  
def compare(exps,variable):
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]
    ax0 = None
    for (i,exp) in enumerate(exps):
        results = pd.read_csv('./experiments/' + exp + '/logs/' + 'results.csv',
                              sep=',',
                              header=0)
        results = results[['Time',variable]].dropna()
        if ax0 is None:
            ax0 = results.plot(x='Time', y=variable,color=colors[i],xlim=(0,50000))
        else:
            results.plot(x='Time', y=variable,color=colors[i],ax=ax0,xlim=(0,50000))
    ax0.legend(exps)
    plt.title(variable + ' through episodes')
    text = variable
    for exp in exps: text+="_"+str(exp)
    plt.savefig(text)
    plt.show()
    

if __name__ == '__main__':
    
    #variables = [Time,State,Epsilon,Action,Reward,Medkit,Poison,Frags,Amo,Max Life,Life,Mean Score,Var Score,Loss]
    exps = []
    for i in range(1,len(sys.argv)):
        exps.append(sys.argv[i])
    compare(exps,'Mean Score')


#python3 process_results.py meas1 meas3
