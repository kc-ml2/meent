function [w,ix,iy]=retsimplifie(w);
%  function [w,ix,iy]=retsimplifie(w);
% eventuelles simplifications d'un maillage de texture 
if length(w)==2;  % 1 D
x=w{1};u=w{2};	
nx=length(x);if nx>1;ix=[find(~all(u(:,2:nx)==u(:,1:nx-1),1)),nx];else ix=1;end;
w={x(ix),u(:,ix)};
	
else;             % 2 D
x=w{1};y=w{2};u=w{3};	
nx=length(x);if nx>1;ix=[find(~all(all(u(2:nx,:,:)==u(1:nx-1,:,:),3),2)).',nx];else ix=1;end;
ny=length(y);if ny>1;iy=[find(~all(all(u(:,2:ny,:)==u(:,1:ny-1,:),3),1)),ny];else iy=1;end;
w={x(ix),y(iy),u(ix,iy,:)};
end;