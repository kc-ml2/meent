function s=retstr(init,t);
%  function s=retstr(init,t);
% matrice S ou G de la translation de vecteur t (ne pas confordre avec rettr translation de l'objet
% translation de vecteur t (uniquement pour les champs) 
% attention a respecter les symetries

n=init{1};beta=init{2};
if init{end}.dim==2; %2D
si=init{8};    
v=repmat(exp(i*t*beta),1,2).';vv=1./v;
if ~isempty(si);vv=diag(si{2}*diag(vv)*si{1});v=diag(si{4}*diag(v)*si{3});end
else               %1D
si=init{3};    
v=exp(i*t*beta).';vv=1./v;
if ~isempty(si);v=diag(si{2}*diag(v)*si{1});vv=diag(si{2}*diag(vv)*si{1});end
end    

s=rets1(init);
if size(s,1)==4;  %  matrice S 
s{1}=diag([vv;v]);
else              %  matrice  G
s{1}=diag([vv;ones(n,1)]);
s{2}=diag([ones(n,1);v]);
end;
