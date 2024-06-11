function a=retprod(varargin);
% function a=retprod(a1,a2,...);
% produit de matrices a1,a2,a3,..  dont certaines sont 
% des cell-array de dimension 2: reel imag pour economiser la memoire
% (eventuellement mis sur fichier par retio )
% le produit est fait dans le sens le plus economique en temps et memoire
% a est un tableau complexe (non sur fichier )


ng=nargin;
ri=0;for ii=1:ng;if iscell(varargin{ii});ri=1;end;end; % ri=1 si une matrice,0 si un cell_array
if ri==1; % real + i * imag
for ii=1:ng;if ~iscell(varargin{ii});varargin{ii}=retio(varargin{ii});varargin{ii}={real(varargin{ii}),imag(varargin{ii})};end;end;
s1=retio(varargin{1}{1},4);s2=retio(varargin{ng}{1},4);
if s1(1)<=s2(2); % produit de gauche a droite
 for ii=2:ng;
 a=varargin{1}{1};varargin{1}{1}=retprodd(a,varargin{ii}{1})-retprodd(varargin{1}{2},varargin{ii}{2});   
 varargin{1}{2}=retprodd(a,varargin{ii}{2})+retprodd(varargin{1}{2},varargin{ii}{1});varargin{ii}=[];
 end;
a=varargin{1}{1}+i*varargin{1}{2};
else;           % produit de droite a gauche   
 for ii=ng-1:-1:1;
 a=varargin{ng}{1};varargin{ng}{1}=retprodd(varargin{ii}{1},a)-retprodd(varargin{ii}{2},varargin{ng}{2});   
 varargin{ng}{2}=retprodd(varargin{ii}{2},a)+retprodd(varargin{ii}{1},varargin{ng}{2});varargin{ii}=[];
 end;
a=retio(varargin{ng}{1})+i*retio(varargin{ng}{2});
end;

else;  % complexes
s1=retio(varargin{1},4);s2=retio(varargin{ng},4);    
if s1(1)<s2(2); % produit de gauche a droite
 a=retio(varargin{1});varargin{1}=[];
 for ii=2:ng;a=a*retio(varargin{ii});varargin{ii}=[];end;
else;            % produit de droite a gauche  
 a=retio(varargin{ng});varargin{ng}=[];
 for ii=ng-1:-1:1;a=retio(varargin{ii})*a;varargin{ii}=[];end;
end;
   
end;   %  ri  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function a=retprodd(a,b);%a=a*b a et b reels;
a=retio(a);b=retio(b);
if ((min(a(:))>-10*eps)&(max(a(:))<10*eps))|((min(b(:))>-10*eps)&(max(b(:))<10*eps));a=spalloc(size(a,1),(size(b,2)),0);return;end;
a=a*b;
