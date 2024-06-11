function [x,y,z,c]=retget(n)
%  [x,y]=retget(n); MANIPULATIONS SUR DES GRAPHIQUES
%  divers possibilites:
%----------------------------------------------------------------------
%  selectionner les points d'une courbe sur un graphe (ou la figure si une seule courbe) puis:
%   ->     [x,y]=retget;  coordonnees des points de la courbe (tracee par  plot) 
%   ->     [x,y,z]=retget;  coordonnees des points de la surface (tracee par mesh ou surface) 
%   ->     [x,y,z,c]=retget;  recuperation des données d'un graphique (trace par pcolor ou retcolor) version P VELHA
%   ->     ou  retget; puis selection d'une figure et copie de la courbe sur cette figure (suivre les indications)
%   ->     ou  [x,y]=retget(-1); permet d'eliminer des points d'une courbe (suivre les indications)
%------------------------------------------------------------------------------
%        retget(1)  transforme un subplot(selectione avant) en une figure entiere
%------------------------------------------------------------------------------
%        retget(n) avec n>=2  pour image obtenue par pcolor
%      creation d'une figure (apres selection)  avec coupe de la fonction  soit sur une ligne (n=2) 
%      soit sur un contour ferme, de n points 
%      ou encore par un nombre indeterminé de points (on arrete en cliquant a droite pour le dernier point (n=0) )  
%

try;

if nargin==0;  %  <------ nargin==0
a=get(gco);Children=a.Children;
for ii=1:length(Children);if ~isempty(findstr(a.Type,'surface'));break;end;a=get(Children(ii));end;
x=a.XData;y=a.YData;z=a.ZData;try;c=a.CData;catch;c=[];end;
if nargout==0;% trace
ch=input('selectionner une figure et entrer le type de ligne( par defaut  ''--r'')');
if isempty(ch);ch='--r';end;
hold on;plot(x,y,ch); 
end;
else;        %  <------ nargin>0
if n==1; %  n=1
%h=gco;figure;set(copyobj(h,gcf),'position',[.13,.11,.775,.815]) 
h=gco;
if ~isempty(h)&&(h~=gcf);figure;set(copyobj(h,gcf),'position',[.13,.11,.775,.815]);else;
g=get(gcf);for h=g.Children.';switch get(h,'Type');case 'axes';figure('color','w');set(copyobj(h,gcf),'position',[.13,.11,.775,.815]);drawnow;end;end; end;   
else;    %  n ~=1

if n<0;% supression de points aberrants
g=gco;a=get(g);Children=a.Children;for ii=1:length(Children);if ~isempty(findstr(a.Type,'surface'));break;end;a=get(Children(ii));end;
x=a.XData;y=a.YData;z=a.ZData;
disp('supression de points clic a droite pour arreter');
iii=1:length(x);while 1;[xx,yy,b]=ginput(1);if b>1;break;end;
[prv,ii]=min((x-xx).^2+(y-yy).^2);iii(ii)=0;
end;
iii=iii(iii>0);x=x(iii);y=y(iii);if ~isempty(z);z=z(iii);end;
set(g,'XData',x,'YData',y,'ZData',z);
return;
end;% n<0


a=get(gco);Children=a.Children;for ii=1:length(Children);if ~isempty(findstr(a.Type,'surface'));break;end;a=get(Children(ii));end;
if n==0;xx=[];yy=[];b=1;disp('au dernier point clic a droite (le circuit est boucle)');while b==1;[xxx,yyy,b]=ginput(1);xx=[xx;xxx];yy=[yy;yyy];end;n=length(xx);
else;[xx,yy]=ginput(n);end;
for ii=1:n;text(xx(ii),yy(ii),['   \leftarrow ', int2str(ii)],'color','k','FontWeight','bold');end;
if n>2;xx=[xx;xx(1)];yy=[yy;yy(1)];n=n+1;end;hold on;plot(xx,yy,'--k','linewidth',3);
x=[];y=[];npt=10000;for ii=1:n-1;x=[x,linspace(xx(ii),xx(ii+1),npt)];y=[y,linspace(yy(ii),yy(ii+1),npt)];end;
[a.XData,nx]=retelimine(a.XData);[a.YData,ny]=retelimine(a.YData);a.CData=a.CData(ny,nx);% elimination des points egaux
z=interp2(a.XData,a.YData,a.CData,x,y);
xd=[0,diff(x)];yd=[0,diff(y)];t=cumsum(sqrt(xd.^2+yd.^2));
figure;hold on;plot(t,z);xlabel('abscisse curviligne');ylabel('fonction');grid;ax=axis;
if n>2;
for ii=1:n-1;plot([t(1+npt*(ii-1)),t(1+npt*(ii-1))],[ax(3),ax(4)-.1*(ax(4)-ax(3))],'--k','linewidth',3);text(t(1+npt*(ii-1)),ax(4)-.05*(ax(4)-ax(3)),int2str(ii),'fontsize',8,'FontWeight','bold');end;
plot([t(end),t(end)],[ax(3),ax(4)-.1*(ax(4)-ax(3))],'--k','linewidth',3);text(t(end),max(z),int2str(1),'FontWeight','bold');
end;
end;       %  n=1  ?
end;      %  <------ nargin

catch;set(gcf,'Pointer','arrow');lasterror,end;