import pandas as pd
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')



if __name__ == '__main__':

    title = sys.argv[1]
    results = pd.read_csv('./experiments/' + title + '/logs/' + 'results.csv',
                          sep=',',
                          header=0)

    results.plot(x='Time', y='Max Life')
    plt.title('Max life through episodes')
    plt.show()

    # results.plot(x='Time', y='Life')
    # plt.title('Life through episodes')
    # plt.show()
