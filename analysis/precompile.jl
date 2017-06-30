
# Call some package
using Ipopt, JuMP, PyPlot, RCall

# Set a working directory

cd("$(homedir())/mixedgauss/CODE/UNCOPT-21")

# input
# n : number of observations, g : grid mult

# output
# L : matrix of likelihood, m : grid size, x : solution, t : computation time
function get_L(n,m)
    @rput n m;
    reval("source('./rebayes.R')");
    L = @rget L;
    x = @rget x;
    t = @rget t;
    return L, x, t
end

# vanilla primal ipopt

function vanilla_ipopt(L)
    n = size(L,1); k = size(L,2);
    m = Model(solver=IpoptSolver(print_level=0));
    @variable(m, x[1:k], start = 1/k);
    @NLobjective(m, Min, -sum(log(sum(L[i,j]*x[j] for j=1:k)) for i=1:n));
    @constraint(m, sum(x) == 1);
    @constraint(m, x.>= 0);
    solve(m);
    return getvalue(x)
end

# line search sequential quadratic programming for primal

function ls_sqp(L)
    n = size(L,1); k = size(L,2);
    iter = 20;
    tol = 1e-2;
    x = ones(k)/k;
    for i = 1:iter
        # gradient and Hessian computation
        Z = Diagonal(1./(L*x)) * L;
        g = -Z'*ones(n);
        H = Z'*Z;
        
        # define a subproblem
        m = Model(solver=IpoptSolver(print_level = 0));
        @variable(m, p[1:k]);
        @objective(m, Min, sum(0.5*p[i]*H[i,j]*p[j] for i = 1:k, j = 1:k )+sum(g[i]*p[i] for i = 1:k));
        @constraint(m, ec, sum(p[i] for i = 1:k) == 0); ## p is only in the simplex
        @constraint(m, ic, p+x .>= 0);                                    
        solve(m);
                                        
        # convergence check - By KKT
        if norm(g - getdual(ic) - getdual(ec),Inf) < tol
            break;
        end
                                        
        # otherwise do update               
        x = abs(getvalue(p)+x); # make sure x is positive
        x = x/sum(x) # renormalize           
    end
    return x
end

# rank_reduced sequential quadratic programming
# tol = 1e-4

function rr_sqp(L)
    n = size(L,1); k = size(L,2);
    F = svdfact(L);
    iter = 20;
    tol = 1e-2;
    x = ones(k)/k;
    
    # svd for rank reduction
    F = svdfact(L);
    ind = F[:S] .> 1e-4;
    U = F[:U][:,ind];
    s = F[:S][ind];
    Vt = F[:Vt][ind,:];
    
    # iteration
    for i = 1:iter
        # gradient and Hessian computation -- Rank reduction method
        d = 1./(U*(Diagonal(s)*(Vt*x)));
        g = -Vt'*(Diagonal(s) * (U'*d));
        H = (Vt'*Diagonal(s)*(U'*Diagonal(d.^2)*U)*Diagonal(s)*Vt);
        
        # define a subproblem
        m = Model(solver=IpoptSolver(print_level = 0));
        @variable(m, p[1:k]);
        @objective(m, Min, sum(0.5*p[i]*H[i,j]*p[j] for i = 1:k, j = 1:k )+sum(g[i]*p[i] for i = 1:k));
        @constraint(m, ec, sum(p[i] for i = 1:k) == 0); ## p is only in the simplex
        @constraint(m, ic, p+x .>= 0);                                    
        solve(m);
                                        
        # convergence check - By KKT
        if norm(g - getdual(ic) - getdual(ec),Inf) < tol
            break;
        end
                                        
        # otherwise do update               
        x = abs(getvalue(p)+x); # make sure x is positive
        x = x/sum(x) # renormalize           
    end
    return x
end

# function evaluation

function eval_f(x) 
  return -sum(log(L*x))
end
