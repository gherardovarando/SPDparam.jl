using LinearAlgebra;
using ForwardDiff;

using StructuredOptimization;

include("src/prxgrd.jl")

#### define problem 
####  argmin ||Ax - b||^2 + lambda * sum(abs(x))
A =       [ 1.0  -2.0   3.0  -4.0  5.0;
            2.0  -1.0   0.0  -1.0  3.0;
           -1.0   0.0   4.0  -3.0  2.0;
           -1.0  -1.0  -1.0   1.0  3.0]
b = [1.0, 2.0, 3.0, 4.0]
lam = (0.05)*norm(A'*b, Inf)
m, n = size(A)
x0 = zeros(Float64, n)
#### 

#### solve with our proximal alg implementation
function f(x::Vector)
        0.5 * norm(A * x - b)^2
end

function g(x)
        lam * norm(x,1)
end

df = x -> ForwardDiff.gradient(f,x)

function pg(x, stp)
         tmp = sign.(x) .* (abs.(x) .- (stp * lam))
         tmp[ abs.(tmp) .<  stp * lam] .= 0.0
         tmp
end

out = prxgrd(f,g,df,pg, x0, 1e-8, 1000)
println("solution with our implementation")
println(out)
println(f(out) + g(out))



### solve with StructuredOptimization 

 
x = Variable(n);

@minimize ls( A*x - b ) + lam*norm(x, 1);


println("solution with StructuredOptimization")
println(~x)
println( f(~x) + g(~x))
