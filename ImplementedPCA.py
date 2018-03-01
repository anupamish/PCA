#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568
import sys
import numpy as np
from numpy import linalg as LA
np.set_printoptions(threshold=np.nan)


def pca(matrix):
    mean = np.mean(matrix,0)
    new_matrix = matrix - mean
    covariance_matrix = np.cov(np.transpose(new_matrix))
    w,v = LA.eig(covariance_matrix)
    U_Truncated = v[:,:2]
    U_Truncated_Transpose = np.transpose(U_Truncated)
    new_dimension_matrix = np.dot(U_Truncated_Transpose,np.transpose(new_matrix))
    newData = np.transpose(new_dimension_matrix)
    val=(-w).argsort()[:2]
    result=v[...,val]
    print("---------------------Direction-----------------------\n")
    print(result.T)
    print("\n---------------------Points--------------------------\n")
    print(newData)

def main():
    input = np.loadtxt(sys.argv[1], dtype='double', delimiter='\t')
    matrix = np.array(input)
    pca(matrix)

if __name__ == '__main__':
    main()