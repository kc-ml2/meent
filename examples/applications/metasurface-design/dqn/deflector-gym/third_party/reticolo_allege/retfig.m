function x=retfig(fg,parm,nrept,nom,cx,clm);
% function x=retfig(fg,parm,nrept,nom,cx,clm);
% fg         7i stockage sous le nom: 100 parm (70000+parm).fig 
% fg        -7i stockage sous le nom: 100 parm (70000+parm).tif 
%                ( et si parm=0 ou absent   cherche un numero non attribue) 
% fg:numero de figure  
% parm:     0  [] ou abs ouverture 
%          >0 stockage sous le nom: 1000000*abs(fg)+abs(parm)  (exemple retfig(1,4) --> 100004n1  )
%                   ( ce qui permet de les retrouver classees par ordre:  7000001 7000002 avant 7000010)
%          -2 impression soignee
%          -1 fichier tif 
%          imaginaire: lecture noir et blanc impression
%  retfig  :fermeture de toutes les figures
%
% si nargin==3: CINEMA
% film avec les figures crees par retfig(fig*i)
% parm:tableau des numeros
% nrept nb de repetitions (3 par defaut)
%  (si nrept=0 ouverture successives des figures sans repetition)
% boucle dans un sens puis dans l'autre nretp fois
% nom si on veut creer un fichier nom.avi (facultatif sinon [])
% cx caxis pour normaliser(facultatif) ou colormap(sous forme de tableau ou chaine de caracteres)
%  si cx a 2 elements:c'est le caxis utilise
%  si cx a un seul element,la dynamique est modifiee
% on utilise alors un colormap special ou bien celui precise dans l'argument clm
%   exemples:
%  retfig:            efface toutes les figures
%  retfig(7):         creation figure 7
%  retfig(7,15):      stocke la figure 7 sous  le nom 70000015.fig
%
%  retfig(5i,15):     stocke la figure OUVERTE sous le nom 50000015.fig
%  retfig(-5i,15):    stocke la figure OUVERTE sous le nom 50000015.tif
%  retfig(7i):        stocke la figure OUVERTE sous le nom 70000143.fig
%  retfig(-7i):       copie la figure OUVERTE dans 70000143.tif (impression soignee)
%
%  retfig(7,-1):      imprime la figure 7
%  retfig(2,15*i):    charge  70000002.fig la met en noir ,l'imprime et l'efface
%  x=retfig(2,-15*i): charge  70000002.fig dans figure(x) 
%  retfig(3,1:16,0);  ouverture de 3000001.fig a 3000016.fig
%
%
% retfig(4,[1:5],2,'film'); cine avec les figures 40000001.fig a 40000005.fig et vis versa repetition 2 fois
% creation de cine.avi
% retfig(2,[1:5],-3); cine avec les figures 20000001.fig a 20000005.fig repetition 3 fois
% retfig(4,[1:5],1,[],'cool'); cine avec les figures 40000001.fig a 40000005.fig colormap 'cool'
% retfig(4,[1:5],1,[],2,'jet'); cine avec les figures 40000001.fig a 40000005.fig colormap 'jet' compression de dynamique

nomfig=inline('int2str(1000000*abs(fg)+abs(parm));','fg','parm');

if nargin>2;% cinema
%cm=interp1(linspace(1,64,11)',[[0,0,0];[0,.25,0];[0,.5,0];[0,.4,.6];[.6,0,.7];[1,0,0];[.9,.6,0];[0,1,0];[1,1,0];[1,1,.5];[1,1,1]],[1:64'],'linear');
%cm=min(1,max(0,cm));
%cm='jet';
if nargin<4;nom=[];end;
if ~isempty(nom);
try;cine=avifile(nom);catch;cine=avifile(nom,'compression','none');end;
end;

parm=parm(:).';if nrept>0;parm=[parm,fliplr(parm)];end;
x=[];for iii=1:max(1,abs(nrept));for ii=parm;
eval(['xx=open(''',nomfig(fg,ii),'.fig'');']);set(xx,'position',get(0,'ScreenSize'));
if nargin>4;
switch(numel(cx));
case 2;for child=get(gcf,'children');set(child,'clim',cx);end;
case 1
if nargin>5;cm=feval(clm);else;
colormap('default');cm=colormap;cm=[[[linspace(0,0.4,8),linspace(0.4,0,8)].',linspace(0,0,16)',linspace(0,1,16)'];cm(9:56,:)];
end;%cm=retcolor;
%load map_jeths2;cm=map_jeths2;
mm=size(cm,1);ccx=linspace(0,1,min(1024,round(mm*cx))).';
ccx=cx*ccx./(1+(cx-1)*ccx);colormap(interp1(1:mm,cm,1+(mm-1)*ccx,'cubic'));
otherwise
colormap(cx);
end;

end;
%fond noir
if nrept~=0
chil=get(gcf,'children');set(gcf,'color',[0,0,0]);
%for chil=get(gcf,'children').';switch get(chil,'Type');case('axes');shading(chil,'interp');set(chil,'color',[0,0,0],'XColor',[1,1,1],'YColor',[1,1,1],'ZColor',[1,1,1]);set(get(chil,'Title'),'Color',[1,1,1]);end;end;
%for chil=get(gcf,'children').';switch get(chil,'Type');case('axes');shading(chil,'interp');set(chil,'color',[0,0,0],'XColor',[1,1,1],'YColor',[1,1,1],'ZColor',[1,1,1]);set(get(chil,'Title'),'Color',[1,1,1]);end;end;
x=[x,xx];if length(x)>1;close(x(end-1));end;
end;

if ~isempty(nom);set(xx,'DoubleBuffer','on');frame=getframe(gcf);cine=addframe(cine,frame);end;

end;end;
if ~isempty(nom);cine=close(cine);end;

return
end;  % fin cinema


x=0;
if nargin<1;close all;return;end;
if nargin<2;parm=[];end;if isempty(parm);parm=0;end;
%ff=int2str(100000+abs(parm));ffg=int2str(100000+abs(fg));fi=[ff,'n',ffg(2:end)];
fi=nomfig(fg,parm);

if imag(fg)>0;
if parm==0;for parm=1:90000;fi=nomfig(fg,parm);if exist([fi,'.fig'],'file')~=2;break;end;end;end;
%if parm==0;for parm=1:90000;fi=[int2str(100000+parm),'n',ffg(2:end)];if exist([fi,'.fig'],'file')~=2;break;end;end;end;
saveas(gcf,fi,'fig');
return;end;


if imag(fg)<0;ffg=abs(fg);fg=floor(ffg);
if parm==0;for parm=1:90000;fi=nomfig(fg,parm);if exist([fi,'.tif'],'file')~=2;break;end;end;end;
%saveas(gcf,fi,'fig');
if fg==ffg;set(gcf,'Renderer','painters');end;
eval(['print -dtiff  -r300 ',fi,'.tif']);
%eval(['print -deps -tiff  -r300 ',fi,'.eps']);
%eval(['print -dpdf -painters -r96 ',fi,'.pdf']);
return;
end;


%if parm==0;if fg==0;figure;else;figure(fg);clf(fg);figure(fg);end;
if parm==0;if fg==0;figure;else;figure(fg);end;
else;
if imag(parm)>0; eval(['x=open(''',fi,'.fig'');colormap(1-gray);print(figure(x));clf(x);']);return;end;% lecture noir et blanc impression
if imag(parm)<0; eval(['x=open(''',fi,'.fig'');']);set(x,'position',get(0,'ScreenSize'));return;end; % lecture
if parm>0;saveas(fg,fi,'fig');return;end;
if parm==-2;set(fg,'Renderer','painters','colormap',1-gray);print(figure(fg));return;end;% impression
if parm==-1;eval(['print(',int2str(fg),',''-dtiff'',''-r200'',''',ffg,'.tif'');']);return,end;% fichier tif
%if parm==-1;eval(['print -dtiff  -r300 ',ffg,'.tif']);return,end;% fichier tif
%else;eval(['print -deps -tiff -r150 ',ffg,'.tif']);end;
%else;eval(['print -dtiff  -r300 'ffg,'.tif']);end;
end;

