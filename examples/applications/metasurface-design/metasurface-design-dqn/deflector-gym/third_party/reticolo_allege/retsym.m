function v=retsym(init,v);
%  function v=retsym(init,v);
%  passage base de fourier base symetrique et vis versa(vecteurs colonnes)



if isempty(init{end}.sym);return;end;
n=init{end}.nsym;m=init{end}.nfourier;

if init{end}.dim==1;  %1 D
if size(v,1)==2*n;  % base symetrique --> base de fourier
if isempty(init{3});return;end;    
v=[init{3}{1}*v(1:n,:);init{3}{1}*v(n+1:2*n,:)];
else;                % base  de fourier --> base  symetrique
v=[init{3}{2}*v(1:m,:);init{3}{2}*v(m+1:2*m,:)];
end; 

else;  %2 D
if isempty(init{8});return;end;   
if size(v,1)==2*n;  % base symetrique --> base de fourier
v=[init{8}{1}*v(1:n,:);init{8}{3}*v(n+1:2*n,:)];
else;                % base  de fourier --> base  symetrique
v=[init{8}{2}*v(1:2*m,:);init{8}{4}*v(2*m+1:4*m,:)];
end; 
end; 
