using LinearAlgebra;
using ForwardDiff;


function gclm_ll(Sigma, B, C, C0, toll = 1e-6, mxitr = 100)
      itr = 0
      delta = 100.0
      fval =
      gval =
      x = x0
      while (delta > toll && itr <= mxitr)
            itr = itr + 1
            grad = df(x0)
            s = 1
            xold = x
            fnew = fval + 1
            gnew = gval
            diff = 0
            while (fnew + gnew > fval + gval || fnew > fval + diff)
                x = pg(xold .- s .* grad, s)
                gnew = g(x)
                fnew = f(x)
                diff =  (1/(2*s)) * sum((x .- xold) .^ 2) +
                         sum( (x .- xold) .* grad)
                s = s * 0.5
            end
            delta = abs(fnew + gnew - fval - gval) / (abs(fval + gval))
            fval = fnew
            gval = gnew
      end
      return x
end
