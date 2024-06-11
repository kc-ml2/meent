function y=rettoeplitz(x,index);
%  function y=rettoeplitz(x); matrice de toeplitz associee au produit par une fonction
%  dont les coefficients de fourier sont ranges dans x de 1 a 2n-1 le coefficient 0 etant x(n)
%  Le resultat est eventuellement mis sous forme de matrice sparse avec retsparse
% 
% Dans le cas d'un usage repete avec une meme taille, il est plus rapide de commencer
% par crer une matrice d'indices et ensuite d'adresser les vecteurs avec cette matrice 
% (c'est ce que fait à chaque passage le programme matlab 'toeplitz' qui est plus lent que le boucle faite ici )
% 
%% Exemple
% m=10;n=500;x=rand(1,2*n-1);ind=rettoeplitz(1:length(x));
% tic;for ii=1:m;a=rettoeplitz(x);end;toc;
% tic;for ii=1:m;a=rettoeplitz(x,ind);end;toc;
% tic;for ii=1:m;a=toeplitz(x);end;toc
%
%% See also; TOEPLITZ RETMAT 

if nargin>1;y=x(index);
else;	
x=x(:);
n=floor((length(x)+1)/2);
y=zeros(n,n);
for ii=1:n;y(:,ii)=x(n-ii+1:2*n-ii);end;
y=retsparse(y);%  
end;
