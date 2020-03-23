using LinearAlgebra;
using ForwardDiff;

TOL = 1e-4
solver = ProximalAlgorithms.ForwardBackward(tol = TOL)

lyap0(B) = lyap(B, 1.0 * Matrix(I, size(B, 1), size(B, 2)))

lam = Float64(0.5)
g = NormL1(lam)

f = SqrNormL2(1.0)

B0 = -1.0 * Matrix(I, 4, 4)

res = solver(B0, f = f, A = I, g = g)

print(res)
