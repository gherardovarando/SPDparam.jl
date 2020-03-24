using ProximalOperators
using ProximalAlgorithms
using LinearAlgebra

A = [  1.0  -2.0   3.0  -4.0  5.0;
            2.0  -1.0   0.0  -1.0  3.0;
           -1.0   0.0   4.0  -3.0  2.0;
           -1.0  -1.0  -1.0   1.0  3.0]
b = [1.0, 2.0, 3.0, 4.0]

m, n = size(A)


lam = (0.1)*norm(A'*b, Inf)

f = Translate(SqrNormL2(1.0), -b)
f2 = LeastSquares(A, b)
g = NormL1(lam)

x_star = [-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

TOL = 1e-4

x0 = zeros(Float64, n)
solver = ProximalAlgorithms.ForwardBackward(tol=TOL)
x, it = solver(x0, f=f, A=A, g=g, L=opnorm(A)^2)
x
