function xhat = omp(Phi,y,K,tolRes)
%implementation of Orthogonal Matching Pursuit (OMP) with stopping
%criterion norm(Phi*xhat-y)<resTol or after K+5 iterations
dbstop error
stopCrit = false;
[M,N] = size(Phi);
xhat = zeros(N,1);
r = y;
Gamma = false(1,N);
l=0;
k=0;
while ~stopCrit
    l=l+1;
    p = Phi'*r;
    [dummy gammaStar] = max(abs(p));
    if ~Gamma(gammaStar)
        Gamma(gammaStar) = true;
        k=k+1;
    end
    PhiGamma = reshape(Phi(repmat(Gamma,M,1)),M,k);
    xGamma = PhiGamma\y;
    r = y-PhiGamma*xGamma;
    xhat(Gamma) = xGamma;
    if norm(r)<tolRes || l==K
        stopCrit = 1;
    end
end
