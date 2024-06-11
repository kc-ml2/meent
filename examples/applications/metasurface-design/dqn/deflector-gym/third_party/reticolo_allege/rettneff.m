function mm=rettneff(init,a,k0,varargin);
%  m=rettneff(init,a,k0,nn,parm,cale,indice_recherche);
% affichage des indices effectifs de la texture de descripteur a 
% puis trace des modes apres choix sur la courbe
% en sortie mm contient les numeros des modes selectionnes 
%
%
% a descripteur de tronçon
% k0=2*pi/ld (1 par defaut)
% nn nombre de points en x et y et eventuellement offset (par defaut [50,50,0,0] en 2D  [100,0] en 1D
% indice_recherche: si present indique la position des modes recherches (facultatif)
%
%
%    en 1 D  si parm ==0, trace de  real et imag(E) ,sinon de abs(E)^2 
%
%    en 2 D
%         nn peut aussi etre {x,y}
%         parm: 0 couleur   1 gris  (par defaut 0)
%         a  peut etre un nom de fichier cree par retio(a,1)
%          si cale est specifie:numeros des composantes des champs a tracer (par defaut les 6)
%         ces champs sont normalises par le max des 6 composantes
%
% See also:RETNEFF,RETTMODE
%
%% Exemples 
%      rettneff(init,a,2*pi/ld,[101,-d/2],2) % 1 D  trace abs(E)^2
%      rettneff(init,a,2*pi/ld,[101,101,-d/2],[],[1:6],1.5) % 2 D centre et pointe l'indice 1.5
%      rettneff(init,a,1,[],0,[1:3],[1.5,1])  % 1 D centre et pointe les indices 1.5 et 1
%      rettneff(init,a,k0,{linspace(-.5,.5,40),linspace(-.5,.5,50)})  % marche pour les cylindres

if nargin<3;k0=[];end;if isempty(k0);k0=1;end;
neff=i*retneff(a)/k0;
fig=figure;hold on;
if nargin>6;
for ii=1:length(varargin{end});
plot(real(varargin{end}(ii)),imag(varargin{end}(ii)),'+g');
text(real(varargin{end}(ii)),imag(varargin{end}(ii)),'   \leftarrow mode cherche','color','g','FontWeight','bold','rotation',90);
end;
varargin=varargin(1:end-1);
end;
plot(real(neff),imag(neff),'.k');grid on;title('neff');xlabel('propagation');ylabel('absorption');
num=0;
while 1;texte={};
n=input('trace de n modes ? entrer n puis choisir sur le graphe (0 pour sortir)  ');if n==0;return;end;
figure(fig);mm=[];
[x,y,prv]=ginput(n);

for ii=1:n;
[prv,kk]=min(abs(neff-x(ii)-i*y(ii)));
mmm=find(abs(neff-neff(kk))<1.e-9)
for jj=1:length(mmm);
mm=[mm,mmm(jj)];
if jj==1;
num=num+1;
plot(real(neff(mmm(jj))),imag(neff(mmm(jj))),'or');text(real(neff(mmm(jj))),imag(neff(mmm(jj))),[' \leftarrow ',int2str(num)],'FontSize',18);
end;
texte=[texte,{rettexte(num)}];
end;end;
if init{end}.genre==2;% pour cylindrique radial
z=linspace(0,init{end}.d(1),101);teta=0;for ii=1:length(mm);
[e,r,wr,o]=retchamp(init,{a},retb(init,a,1,1,[],[]),retb(init,a,-1,1,mm(ii),[]),1,{z,teta},[[0,1,1];[1,1,0]],[],[1:6,i]);
rettchamp(e,o,z,teta,r,[1:6,i],[],[],rettexte(mm(ii)));    
end   
    
else;rettmode(init,a,{mm,texte},varargin{:});end;
disp(rettexte('numeros des modes selectionnes  ',mm));
%mm=[mm,mmm];
end;
