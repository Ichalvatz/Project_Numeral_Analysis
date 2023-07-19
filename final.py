
import numpy as np
from numpy.polynomial import Polynomial
from numpy.linalg import lstsq
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import qr
import matplotlib.pyplot as plt


#DImiourgia toy t dianismatos
t = [0]
for i in range(1, 50):
    t.append(i / 50)
t = np.array(t)

#dimiourgia y
y = np.cos(4 * t) + 0.1 * np.random.randn(t.shape[0])

#dimourgia tou Vandermonde matrix gia tin methodo twn elaxiston tetragwnwn thn paragontopoihsh LU kai QR
deg = 4
Vandermonde_matrix = np.vander(t, deg +1, increasing=True)
print(Vandermonde_matrix)

#METHODOS TWN ELAXISTVN TETRAGONWN
coefficient, residuals, rank, singular_values = lstsq(Vandermonde_matrix, y)


poly_1 = Polynomial(coefficient)

errors_LS = y - poly_1(t)
square_errors_LS = np.sum(errors_LS**2)
print("Prossegish gia methodo elaxiston tetragonon : ")
print(poly_1)


#PARAGONTOPOIHSH LU
A = Vandermonde_matrix .T @ Vandermonde_matrix   
b = Vandermonde_matrix .T @ y  

LU, piv = lu_factor(A)  

coefficient2 = lu_solve((LU, piv), b)  

poly_2 = np.poly1d(coefficient2[::-1])  

errors_LU = y - poly_2(t)
square_errors_LU = np.sum(errors_LU**2)
print("Prossegish me thn paragontopoihsh LU :")
print(poly_2)

#PARAGONTOPOIHSH QR
Q, R = qr(Vandermonde_matrix)  

QTY = Q.T @ y  

coeffisient3 = np.linalg.solve(R[:deg + 1, :], QTY[:deg + 1])  

poly_3 = np.poly1d(coeffisient3[::-1])  

errors_QR = y - poly_3(t)
square_errors_QR = np.sum(errors_QR**2)
print("Prossegish me paragontopoihsh QR :")
print(poly_3)


print("Athroisma ton tetragonikon sfalmataon:")
print("Paragontopoihsh LU:", square_errors_LU)
print("Paragontopoihsh QR:", square_errors_QR)
print("Methoso elaxiston tetragonon :", square_errors_LS)

# Kataskeyi tou diagrammatos
plt.scatter(t, y,10,'black', label='Data Points')
plt.plot(t, poly_2(t),'b--', label='LU Factorization',linewidth = 3)
plt.plot(t, poly_3(t),'r-', label='QR Factorization',linewidth = 1.5)
plt.plot(t, poly_1(t),'g:', label='Least Squares', linewidth = 3.5)
plt.title('Approximations')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()



# prosthiki twn SSE timwn sto diagrama
text = f'SSE (LU Factorization): {square_errors_LU:.4f}\nSSE (QR Factorization): {square_errors_QR:.4f}\nSSE (Least Squares): {square_errors_LS:.4f}'
plt.text(0.7, 1.1, text, transform=plt.gca().transAxes)

# emfanisi diagramatos
plt.show()