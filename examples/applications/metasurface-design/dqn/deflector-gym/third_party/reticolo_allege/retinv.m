function [a,b]=retinv(a,b);
% [aa,bb]=retinv(a,b);
%  aa+i*bb=inv(a+i*b)

if norm(a)>norm(b);
aa=b*inv(a);
%aa=(a.'\b.').';
a=inv(a+aa*b);b=-a*aa;
else;
bb=-a*inv(b);
%bb=-(b.'\a.').';;
b=inv(-b+bb*a);a=b*bb;
end;






