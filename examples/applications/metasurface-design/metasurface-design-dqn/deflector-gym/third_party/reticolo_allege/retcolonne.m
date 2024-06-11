function x=retcolonne(x,parm);
% x=retcolonne(x)    x=x(:);
% x=retcolonne(x,1)  x=x(:).';
% permet d'utiliser l'ordre dans une expression sans utiliser une matrice intermediaire 
% par exemple: sum(retcolonne(sqrt(a)).*retcolonne(sqrt(b)));
if nargin<2;parm=0;end;
x=x(:);
if parm==1;x=x.';end;