function rettmode(init,a,m,nn,parm,cale);
% function rettmode(init,a,m,nn,parm,cale);
% trace des modes  de numeros m, a calcule par retcouche
% nn nombre de points en x et y et eventuellement offset (par defaut [50,50,0,0] en 2D  [100,0] en 1D
%
%    en 1 D  si parm ==0, trace de  real et imag(E) ,sinon de abs(E)^2 
%          nn peut aussi etre x et doit contenir au moins 3 points
%
%    en 2 D
%         nn peut aussi etre {x,y}
%         parm: 0 couleur   1 gris  (par defaut 0)
%         a  peut etre un nom de fichier cree par retio(a,1)
%          si cale est specifie:numeros des composantes des champs a tracer (par defaut les 6)
%         ces champs sont normalises par le max des 6 composantes
%
%     si m est un cell array de taille [1,2]
%     m{1} est le tableau des numeros, m{2} un cell array de commentaires(meme longueur que m{1})                         = un  
%    qui sera imprime avec le trace du mode (par exmple un numero ..)
%
% Il est preferable d'utiliser rettneff
%
% See also: RETTNEFF RETMODE 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%if a{end}.type==2;  % metaux
%sh=retb(init,a,1.e-3);sh=rettronc(sh,inc,[],1);sb=retb(init,a,-1.e-3);sb=rettronc(sb,[],[],-1); 
%e=retmchamp(init,a,sh,sb,inc,x,0,1);
%end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

if iscell(m);texte=m{2};m=m{1};else;texte=cell(1,length(m));end;
a=retio(a);

if init{end}.dim==2; %2D
if nargin<6;cale=[1:6];end;
if nargin<4;nn=[];end;

if init{end}.genre==1; % cylindres Popov
if isempty(nn);nn=2*init{end}.d;end;
if ~iscell(nn);nn={linspace(-nn(1),nn(1),101),linspace(-nn(1),nn(1),101)};end;
else;
if isempty(nn)|(~iscell(nn)&nn==0);nn=[50,50];end;
end;
if nargin<5;parm=0;end;

sog=init{end}.sog;
dd=init{end}.d;n=init{1};mm=length(m);
if iscell(nn);x=nn{1};y=nn{2};else;% nn contient les x et y
if length(nn)==4;x0=nn(3);y0=nn(4);else x0=0;y0=0;end;
x=x0+linspace(0,dd(1),nn(1));y=y0+linspace(0,dd(2),nn(2));
end;

if a{end}.type==2;  % metaux
d=a{3};
e=[];for ii=1:mm;e=cat(1,e,retchamp(init,a,m(ii),[],[],{x,y},0,0));end;% forme acceleree de retchamp
else;  % dielectriques
d=a{5};
inc=full(sparse(m,[1:mm],1,2*n,mm,mm));e=retchamp(init,a,inc,1,{x,y});
end;
e=permute(abs(e).^2,[1,3,2,4]);



text=['EX';'EY';'EZ';'HX';'HY';'HZ'];
if length(cale)==1;p1=1;p2=1;end;% pour retsubplot
if length(cale)==2;p1=1;p2=2;end;
if length(cale)==3;p1=2;p2=2;end;
if length(cale)==4;p1=2;p2=2;end;
if length(cale)==5;p1=2;p2=3;end;
if length(cale)==6;p1=2;p2=3;end;
for k=1:mm;
figure;
a=max(max(max(abs(e(k,:,:,:)))));
kkk=0;
for kk=cale;kkk=kkk+1;
retsubplot(p1,p2,kkk);retcolor(x,y,e(k,:,:,kk)/a,parm);xlabel('x');ylabel('y');
if kkk==1;title([texte{k},'bet=',num2str(d(m(k))),'  ',text(kk,:)],'fontsize',8);else;title(text(kk,:));end;  
end;

drawnow;
end

else; %1D
if nargin<4;nn=[];end;if isempty(nn)|nn==0;nn=100;end;
if nargin<5;parm=0;end;
n=init{1};dd=init{end}.d;mm=length(m);
if length(nn)>2;x=nn;else;if length(nn)==2;x0=nn(2);else x0=0;end;x=x0+linspace(0,dd,nn(1));end;
if a{end}.type==2;  % metaux
d=a{3};
% sh=retb(init,a,1.e-3);sb=retb(init,a,-1.e-3);sb=rettronc(sb,[],[],-1); 
% e=[];for ii=1:mm;e=cat(1,e,retchamp(init,a,rettronc(sh,m(ii),[],1),sb,1,x,0,1));end;
e=[];for ii=1:mm;e=cat(1,e,retchamp(init,a,m(ii),[],[],x,0,0));end;% forme acceleree de retchamp
else;  % dielectriques
d=a{5};
inc=full(sparse(m,[1:mm],1,2*n,mm,mm));e=retchamp(init,a,inc,1,x);
end;
nnn=ceil(sqrt(mm));
figure;
if parm==0;
for ii=1:mm;retsubplot(nnn,nnn,ii);plot (x,real(e(ii,:,1)),x,imag(e(ii,:,1)));title(num2str(d(m(ii))));end;
else;
for ii=1:mm;retsubplot(nnn,nnn,ii);plot (x,abs(e(ii,:,1)).^2);title(num2str(d(m(ii))));end;
end;
end;











