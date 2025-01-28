import numpy
_A = [[1,2,3], [4,5,6], [7,8,9]] #it just an index right here, or list of an element...
#using numpy as np.array for make it to a vector or an matrix
A = numpy.array(_A)
print(_A, "\n") #print index _A
print(A) #print matrix A

# index an matrix by using row index and using collom index, just like access an element in C++ with the same syntax
print("A[2[2] is: " , A[2][2])
 # notice that index in array in python always start at index 0

#arithmetic in array and vector

B = numpy.array([[1,3,2],[2,4,1],[3,2,5]])
print("matrix B is: \n", B)
# +
print("matrix A+B is: \n", A+B)
# -
print("matrix A-B is: \n" ,A-B)
# duplicate and diverse for a number(s)
x= 10
print("key of A*x: \n", A*x)
print("key of A/x: \n", A/x)

#with matrix and duplicate, notice that matrix just can be dup by the same matrix under: nxp * pxm => n*m
# A is matrix of level 3
C = numpy.array([[1],[2],[3]])
print("matrix A*B: \n", A*B) #difference, meaning that A[i][j] * B[i][j] not dup matrix with matrix
print("matrix A*C: \n", A.dot(C))

#basis of R^n by using eyes

D = numpy.eye(4)
print("basis of R^4 is: \n", D)
#matrix of boolean:
print(D==0) #using print with check var if any element is 0 => true
print(D==1) #using print with check var if any element is 1 => true
print(D==2) #=> false all matrix

# find det of matrix 

Inverse_matrix = numpy.array([[1,3,2],[2,1,4],[3,2,1]])
print("det of matrix Inverse_matrix is: ", numpy.linalg.det(Inverse_matrix))
if numpy.linalg.det(Inverse_matrix) == 0 :
    print("matrix Inverse_matrix do not linear independence and A is degenerate matrix")
else: 
    print("matrix Inverse_matrix is linear independence")
    print("inverse matrix More-penrose", numpy.linalg.pinv(Inverse_matrix)) # that's inverse matrix more penrose, meaning that not the exact inverse matrix, just at lim of this matrix
    print("Inverse matrix exact: ", numpy.linalg.inv(Inverse_matrix))
    print("basis of matrix follow more-penrose is: \n", Inverse_matrix@numpy.linalg.pinv(Inverse_matrix))
    print("basis of matrix is: \n", Inverse_matrix@numpy.linalg.inv(Inverse_matrix))


#transpose matrix

print("transpose of matrix A is: \n", numpy.transpose(A))

#size function with matrix
print("size of matrix A is: ", numpy.size(A))
print("size of row of A is: ", numpy.size(A,0))
#axis: 0 is row, 1 is collom
#with some syntax of matrix using axis for direction, if no meaning, usually used in all matrix
#max min and sum of matrix function

print("sum of matrix A of row is: \n", numpy.sum(A,0))
print("max of matrix A is: ", numpy.max(A))
print("min of matrix A is:", numpy.min(A))