function out = randpermk(n,k)
%same as randperm in Matlab2012, which is compatible with Matlab2010
%(Matlab built-in function randperm(n,k) is not downwards compatible)
%
%  P = randpermk(N,K) returns a row vector containing K unique integers
%  selected randomly from 1:N.  For example, randperm(6,3) might be [4 2 5].

helper = randperm(n);
if nargin == 1
    out = helper;
else
    out = helper(1:k);
end

