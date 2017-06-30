
# setting : gridmult is fixed by m = 2
# n_max = 30 * 2^10 > 30000

iter = 10;
range = 30 * 2.^(1:iter);
m = 1.5;
t1 = zeros(iter); t2 = zeros(iter); t3 = zeros(iter);# t4 = zeros(iter);
d1 = zeros(iter); d2 = zeros(iter);# d3 = zeros(iter);
f1 = zeros(iter); f2 = zeros(iter);# f3 = zeros(iter);

# I don't know why this is needed, but error occurs otherwise
n = 20;
@rput n m;
reval("source('./rebayes.R')");
L = @rget L;
k = size(L,2);

# compute time
for i = 1:iter
    n = range[i];
    @rput n m;
    reval("source('./rebayes.R')");
    L = @rget L;
    x3 = @rget x
    t3[i] = @rget t; # t3 : RMosek
    tic();
    x2 = ls_sqp(L);
    t2[i] = toq(); # t2 : Line search SQP
    tic();
    x1 = vanilla_ipopt(L); # t1 : Vanilla_IPOPT
    t1[i] = toq();
    f1[i] = eval_f(x2)/eval_f(x1) - 1;
    f2[i] = eval_f(x3)/eval_f(x1) - 1;
    d1[i] = norm(x2-x1,1);
    d2[i] = norm(x3-x1,1);
end

# make time plot
# scale = log2
fig, ax = subplots()
ax[:plot](log2(range),log2(t1),label="Vanilla IPOPT");
ax[:plot](log2(range),log2(t2),label="Line Search SQP");
ax[:plot](log2(range),log2(t3),label="Ashr - RMosek");
ax[:legend](loc="best")
xlabel("log2(n)");
ylabel("log2(time)");
title("Computation time, k = $size(L,2)");

# make function value plot

fig, ax2 = subplots()
ax2[:plot](log2(range),f1,label="Line search - Vanilla");
ax2[:plot](log2(range),f2,label="RMosek - Vanilla");
ax2[:legend](loc="best")
xlabel("log2(n)");
ylabel("relative difference");
title("Relative difference between optimal values");

# make l1 distance plot

fig, ax3 = subplots()
ax3[:plot](log2(range),d1,label="Line search - Vanilla");
ax3[:plot](log2(range),d2,label="RMosek - Vanilla");
ax3[:legend](loc="best")
xlabel("log(n)");
ylabel("L1 distance");
title("L1 distance between solutions");
