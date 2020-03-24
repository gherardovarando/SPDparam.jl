using LinearAlgebra;
using ForwardDiff;

include("src/prxgrd.jl")


####  argmin ||Ax - b||^2 + lambda * sum(abs(x))
lambda = 10
A = [ 1.0  -2.0   3.0  -4.0  5.0;
            2.0  -1.0   0.0  -1.0  3.0;
           -1.0   0.0   4.0  -3.0  2.0;
           -1.0  -1.0  -1.0   1.0  3.0]
b = [1.0, 2.0, 3.0, 4.0]

function ff(x::Vector)
        sum((A * x .- b) .^2 )
end

function gg(x)
        lambda * sum(abs.(x))
end

df = x -> ForwardDiff.gradient(ff,x)

function pg(x, step)
         out = sign.(x) .* (abs.(x) .- (step * lambda))
         out[ abs.(out) .<  step * lambda] .= 0.0
         out
end

x0 = zeros(Float64, 5)
df(x0)
out = prxgrd(ff,gg,df,pg, x0, 1e-8, 1000)
out
ff(out)
gg(out)
