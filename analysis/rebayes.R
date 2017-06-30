library("REBayes")
set.seed(100)
n = floor(n/3)
z = c(rt(n,df=2),2*rt(n,df=4),4*rt(n,df=6))
z = z[order(abs(z))]
data = ashr::set_data(z,1)
grid = ashr:::autoselect.mixsd(data, mult=m, mode=0)
grid = c(0,grid)
k = length(grid)
g  = ashr::normalmix(rep(1/k,k),rep(0,k),grid)
llik <- t(ashr:::log_comp_dens_conv.normalmix(g,data))
L = llik - apply(llik, 1, max)
L = exp(L)
t = system.time(res <- REBayes::KWDual(L, rep(1,k), rep(1,n)/n))[3]
min.f = 1E-8
res$f[res$f < min.f] = 0
x = res$f / sum(res$f)