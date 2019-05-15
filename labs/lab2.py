
import pandas as pd
import numpy as np

def v_opt_dp(x, num_bins):
    
    matrix1=[]
    
    for i in range(num_bins):
        matrix2=[]
        for j in range(len(x)):
            matrix2.append(-1)
        matrix1.append(matrix2)
    
    matrix_index=[]

    for i in range(num_bins):
        i1=[]
        for j in range(len(x)):
            i1.append(-1)
        matrix_index.append(i1)

    recursion(0, num_bins - 1, x, num_bins, matrix1, matrix_index)

    begin = matrix_index[-1][0]
    forward = begin
    bins = [x[:begin]]
    
    up_limit = len(matrix_index)-2
    while(up_limit>0):
        begin = matrix_index[up_limit][begin]
        bins.append(x[forward:begin])
        forward = begin
        up_limit-=1
        
 
    bins.append(x[forward:])
    return matrix1, bins


def recursion(rec_mat, rest, x, num_bins, matrix1, matrix_index):
    sub = num_bins - rest - rec_mat
    sub1 = len(x) - rec_mat
    a=[]
    if (sub < 2) and (sub1 > rest):
        recursion(rec_mat + 1, rest, x, num_bins, matrix1, matrix_index)
        if (rest == 0):
            matrix1[rest][rec_mat] = np.var(x[rec_mat:]) * len(x[rec_mat:])
            return
        recursion(rec_mat, rest - 1, x, num_bins, matrix1, matrix_index)
        mtx_l = [matrix1[rest - 1][rec_mat + 1]]

##        for i in range(rec_mat+2, len(x)):
##            a.append(matrix1[rest - 1][i] + (i - rec_mat) * np.var(x[rec_mat:i]))
##            mtx_l.extend(a)
        
        mtx_l.extend( [matrix1[rest - 1][i] + (i - rec_mat) * np.var(x[rec_mat:i]) for i in range(rec_mat + 2, len(x))])

        matrix1[rest][rec_mat]=mtx_l[0]
        
        for i in range(len(mtx_l)):
            if matrix1[rest][rec_mat]>mtx_l[i]:
                matrix1[rest][rec_mat]=mtx_l[i]
        
        
        matrix_index[rest][rec_mat] = mtx_l.index(min(mtx_l)) + rec_mat + 1



