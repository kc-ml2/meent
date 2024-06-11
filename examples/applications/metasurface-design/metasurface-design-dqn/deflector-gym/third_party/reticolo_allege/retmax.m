function [am,im,m]=retmax(a);
% function [am,im,m]=retmax(a);max d'un tableau a plusieurs dimensions
% am:max =a(im(1),im(2)   )=a(m)
s=size(a);
[am,m]=max(a(:));
im=[];
n=length(s);
k=[1,cumprod(s(1:end-1))];
mm=m-1;
for jj=n:-1:1,
im=[floor(mm/k(jj))+1,im];
mm=rem(mm,k(jj));
end;
