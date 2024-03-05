import numpy as np
from scipy.linalg.blas import dgemm
import io
import sys

def pad_pair(A,B): #m of A is != n of B, m/n is not even
    if len(A)%2:
        A = np.concatenate((A,np.zeros([1,len(A[0])],dtype=int)),axis=0)
    if len(A[0])%2:
        A = np.concatenate((A,np.zeros([len(A),1],dtype=int)),axis=1)
    if len(B)%2:
        B = np.concatenate((B,np.zeros([1,len(B[0])],dtype=int)),axis=0)
    if len(B[0])%2:
        B = np.concatenate((B,np.zeros([len(B),1],dtype=int)),axis=1)
    """
    if len(A[0])<len(B):
        A = np.concatenate((A,np.zeros([len(A),len(B)-len(A[0])],dtype=int)),axis=1)
    elif len(A[0])>len(B):
        B = np.concatenate((B,np.zeros([len(A[0])-len(B),len(B[0])],dtype=int)),axis=0)
    """
    return A,B
        

def remove_pad(a,row_n,col_n):
    a = a[:row_n]
    a = np.transpose(a)
    a = a[:col_n]
    a = np.transpose(a)
    return a

def strassen(a,b,lvl=0):
    orig_row_num = len(a)
    orig_col_num = len(b[0])
    A,B = pad_pair(a,b)
    if lvl==2: #GEMM (directly/without splitting)
        C = dgemm(1,A,B)
    else: #go down 1 strassen recursion level
        lvlup = lvl+1
        A11, A12, A21, A22 = A[:int(len(A)/2),:int(len(A[0])/2)], A[:int(len(A)/2),int(len(A[0])/2):], A[int(len(A)/2):,:int(len(A[0])/2)], A[int(len(A)/2):,int(len(A[0])/2):]
        B11, B12, B21, B22 = B[:int(len(B)/2),:int(len(B[0])/2)], B[:int(len(B)/2),int(len(B[0])/2):], B[int(len(B)/2):,:int(len(B[0])/2)], B[int(len(B)/2):,int(len(B[0])/2):]

        X = A11-A21
        Y = B22-B12
        C21 = strassen(X,Y,lvlup)
        X = A21+A22
        Y = B12-B11
        C22 = strassen(X,Y,lvlup)
        X = X-A11
        Y = B22-Y
        C12 = strassen(X,Y,lvlup)
        X = A12-X
        C11 = strassen(X,B22,lvlup)
        X = strassen(A11,B11,lvlup)
        C12 = X+C12
        C21 = C12+C21
        C12 = C12+C22
        C22 = C21+C22
        C12 = C12+C11
        Y = Y-B21
        C11 = strassen(A22,Y,lvlup)
        C21 = C21-C11
        C11 = strassen(A12,B21,lvlup)
        C11 = X+C11

        C = np.concatenate((C11,C12),axis=1)
        temp = np.concatenate((C21,C22),axis=1)
        C = np.concatenate((C,temp),axis=0)

    return remove_pad(C,orig_row_num,orig_col_num)


def main(file):
    f = io.open(file,"r")
    o = io.open("se03_out_{}".format(file[8:len(file)]), "w")
    t_num = int(f.readline())
    t_perm = t_num
    
    while(t_num):
        dim = np.array(f.readline().split(),dtype=int)
        a_row, a_col, b_row, b_col = dim[0], dim[1], dim[2], dim[3]

        a = np.zeros((a_row,a_col))
        b = np.zeros((b_row,b_col))
        for i in range(a_row):
            a[i] = np.array(f.readline().split(),dtype=int)
        for i in range(b_row):
            b[i] = np.array(f.readline().split(),dtype=int)

        C = strassen(a,b)
        C = C.astype('int')
        #print(C)

        o.write(f"Case #{t_perm-t_num+1}:\n")
        o.write(f"{a_row} {b_col}\n")
        for i in range(a_row):
            for j in range(b_col):
                o.write(str(C[i][j]))
                o.write(" ")
            o.write('\n')

        t_num -= 1

    f.close()
    o.close()

input_file = sys.argv[1]
main(input_file)
