function c=retslash(a,b);
% c=a/b aprés preconditionnement
pprv=retdiag(1./max(abs(b),[],2));b=pprv*b;
prv=retdiag(1./max(abs(b),[],1));b=b*prv;
c=((a*prv)/b)*pprv;


