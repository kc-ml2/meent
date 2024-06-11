function c=retantislash(a,b);
% c=a\b aprés preconditionnement
prv=retdiag(1./max(abs(a),[],1));a=a*prv;
pprv=retdiag(1./max(abs(a),[],2));a=pprv*a;
%c=(1./prv).'.*(a\((1./pprv).*b));
c=prv*(a\(pprv*b));
