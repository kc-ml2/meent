function [p,f,ff]=retbornes(p,box,marges);if isempty(p);f=zeros(0,1);ff=zeros(0,1);return;end;
% function [pp,f,ff]=retbornes(p,box,marges);
% troncation d'un vecteur p a box avec des marges
% pp=p(f,:) p(ff,:) sont elimines
% si marges est complexe,on projette les points dans les marges sur les bords de box
% p et pp sont des vecteurs de size [nb points ,dim]
% size(box) est [2,dim]  (angles extremes)
% marges est soit un scalaire,soit un vecteur de size [1,2],soit un vecteur de size [2,2]
% (dans les 2 premiers cas marge est complete par defaut marge=0)
% box(1,:) devient box(1,:)+marges(1,:)
% box(2,:) devient box(2,:)-marges(2,:)
%  plus marge est grand plus on elimine de points (marge peut etre<0)

dim=size(box,2);if nargin<3;marges=0;end;


if any(marges~=0);
if length(marges(:))==1;marges=marges*ones(2,dim);end;
if size(marges,1)==1;marges=[marges;marges];end;

if ~isreal(marges);% projection sur box
marges=real(marges);
for ii=1:dim;
p( (p(:,ii)>(box(1,ii)+marges(1,ii))) & (p(:,ii)<(box(1,ii))) ,ii)=box(1,ii);
p( (p(:,ii)<(box(2,ii)-marges(2,ii))) & (p(:,ii)>(box(2,ii))) ,ii)=box(2,ii);
end;
end;
box(1,:)=box(1,:)+marges(1,:);
box(2,:)=box(2,:)-marges(2,:);
end;
% 
% f=find((p(:,1)>=box(1,1))&(p(:,1)<=box(2,1)));
% for k=2:dim;
% f=f(find((p(f,k)>=box(1,k))&(p(f,k)<=box(2,k))));
% end;

f=(p(:,1)>=box(1,1))&(p(:,1)<=box(2,1));
for k=2:dim;
f(f)=(p(f,k)>=box(1,k))&(p(f,k)<=box(2,k));
end;

if nargout>2;[f,ff]=retfind(f);else;f=find(f);end;

%if nargout>2;ff=setdiff([1:size(p,1)].',f);end;
p=p(f,:);
