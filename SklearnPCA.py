#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import sys
import numpy as np
from sklearn.decomposition import PCA,IncrementalPCA
np.set_printoptions(threshold=np.nan) #To print all the rows. 

def main():
    input = np.loadtxt(sys.argv[1], dtype='double', delimiter='\t')
    matrix = np.array(input)
    ipca = IncrementalPCA(n_components=2, batch_size=10)
    X_ipca = ipca.fit_transform(matrix)
    print(X_ipca)


if __name__ == '__main__':
    main()
