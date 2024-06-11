function [c,h]=retcolor(x,y,a,parm,pparm);
% retcolor(x,y,a,parm,pparm) ou retcolor(a,parm) ou c=retcolor(parm)
% idem a pcolor mais sans oublier la dernière ligne et la dernière colonne
% Attention: la valeur finale de x et y est modifièe pour afficher le dernier point
%  (si x et y ne sont pas ordonnes mise en ordre)
% parm: < 0: pas de colorbar si abs(parm)==1:gris
% parm: 2*i :mesh 
% parm: -n*i :contours avec n lignes  (n non entier pas de clabel)
% parm: [-i,v[:]]  :contours avec le vecteur v
% parm: 3i anaglyphes (stero en test.. ) 
%
% c=retcolor(parm) pour changer le colormap parm=0: jet mais ne cree pas de figure ..
% parm=0 jet    2 gray    3 1-gray
%
% pparm=  caxis de 0 a pparm (pour fonctions >=0)
%
%
% pour des fonctions complexes de variable complexe  z-->Z
%      retcolor(z,Z,parm,nf) nf=1 1 seule figure  sinon 2 ou 4 figures (défaut)
%      parm  même rôle que pour les autres traces   
%      si possible(sauf pour pcolor et contours)  trace aussi la fonction inverse (Z-->z) 
% %exemples
%  x=linspace(-1,1,51);y=linspace(-2,2,51);
%  [X,Y]=meshgrid(x,y);a=X.^2-Y.^2;
%  figure;subplot(2,3,1);retcolor(X,y,a);title('retcolor(X,y,a)');
%  subplot(2,3,2);retcolor(x,y,a,-2);title('retcolor(x,y,a,-2)');
%  subplot(2,3,3);retcolor(x,y,a,2i);title('retcolor(x,y,a,2i)');
%  subplot(2,3,4);retcolor(x,y,a,-20.1i);title('retcolor(x,y,a,-20.1i)');
%  subplot(2,3,5);A=rand(3,3);retcolor(A);title('retcolor(A)');
%  subplot(2,3,6);pcolor(A);title('pcolor(A)');
%  figure;retcolor(x,y,a,3i);title('retcolor(x,y,a,3i)');
%  figure;retcolor(X(:),Y(:),a(:),3i);title('retcolor(x,y,a,3i)');
%  figure;[x,y]=meshgrid(linspace(-1,1,41),linspace(-1,1,41));z=x+i*y;Z=z.^2;retcolor(z,Z,2i,1);
%
% See also: PCOLOR,MESH,CONTOURS,COLORMAP,COLORMAPEDITOR,RETTCHAMP
 
% fonction complexe de variable complexe x --> y ---------------------------------------------------------------------------------------
 


if nargout==1;  % colormap
if nargin==0;x=1;end; % par defaut
switch(x);
case 0;% jet
u =[(1:16)/16,ones(1,15),(16:-1:1)/16].';c=zeros(64,3);c(25:64,1)=u(1:40);c(9:55,2)=u(1:47);c(1:39,3)=u(9:47);
case 2;c=repmat(linspace(0,1,64).',1,3);%gray
case 3;c=repmat(linspace(1,0,64).',1,3);%1-gray

case 1;

   c=[      0            0  5.0196e-001;            0            0  5.9085e-001;            0            0  6.7974e-001;
            0            0  7.6863e-001;  6.9935e-002            0  7.8105e-001;  1.3987e-001            0  7.9346e-001;
  2.0980e-001            0  8.0588e-001;  2.7974e-001            0  8.1830e-001;  3.4967e-001            0  8.3072e-001;
  4.1961e-001            0  8.4314e-001;  4.8123e-001            0  7.2269e-001;  5.4286e-001            0  6.0224e-001;
  6.0448e-001            0  4.8179e-001;  6.6611e-001            0  3.6134e-001;  7.2773e-001            0  2.4090e-001;
  7.8936e-001            0  1.2045e-001;  8.5098e-001            0            0;  8.7582e-001  1.8301e-002  1.8301e-002;
  9.0065e-001  3.6601e-002  3.6601e-002;  9.2549e-001  5.4902e-002  5.4902e-002;  9.5033e-001  7.3203e-002  7.3203e-002;
  9.7516e-001  9.1503e-002  9.1503e-002;  1.0000e+000  1.0980e-001  1.0980e-001;  9.8752e-001  1.5080e-001  1.0018e-001;
  9.7504e-001  1.9180e-001  9.0553e-002;  9.6257e-001  2.3280e-001  8.0927e-002;  9.5009e-001  2.7380e-001  7.1301e-002;
  9.3761e-001  3.1480e-001  6.1676e-002;  9.2513e-001  3.5579e-001  5.2050e-002;  9.1266e-001  3.9679e-001  4.2424e-002;
  9.0018e-001  4.3779e-001  3.2799e-002;  8.8770e-001  4.7879e-001  2.3173e-002;  8.7522e-001  5.1979e-001  1.3547e-002;
  8.6275e-001  5.6078e-001  3.9216e-003;  8.4471e-001  6.0863e-001  4.3137e-002;  8.2667e-001  6.5647e-001  8.2353e-002;
  8.0863e-001  7.0431e-001  1.2157e-001;  7.9059e-001  7.5216e-001  1.6078e-001;  7.7255e-001  8.0000e-001  2.0000e-001;
  7.1709e-001  8.2857e-001  2.2633e-001;  6.6162e-001  8.5714e-001  2.5266e-001;  6.0616e-001  8.8571e-001  2.7899e-001;
  5.5070e-001  9.1429e-001  3.0532e-001;  4.9524e-001  9.4286e-001  3.3165e-001;  4.3978e-001  9.7143e-001  3.5798e-001;
  3.8431e-001  1.0000e+000  3.8431e-001;  4.4588e-001  1.0000e+000  3.4588e-001;  5.0745e-001  1.0000e+000  3.0745e-001;
  5.6902e-001  1.0000e+000  2.6902e-001;  6.3059e-001  1.0000e+000  2.3059e-001;  6.9216e-001  1.0000e+000  1.9216e-001;
  7.5373e-001  1.0000e+000  1.5373e-001;  8.1529e-001  1.0000e+000  1.1529e-001;  8.7686e-001  1.0000e+000  7.6863e-002;
  9.3843e-001  1.0000e+000  3.8431e-002;  1.0000e+000  1.0000e+000            0;  1.0000e+000  1.0000e+000  9.4608e-002;
  1.0000e+000  1.0000e+000  1.8922e-001;  1.0000e+000  1.0000e+000  2.8382e-001;  1.0000e+000  1.0000e+000  3.7843e-001;
  1.0000e+000  1.0000e+000  4.7304e-001;  1.0000e+000  1.0000e+000  5.6765e-001;  1.0000e+000  1.0000e+000  6.6225e-001;
  1.0000e+000  1.0000e+000  7.5686e-001];



end;
return;
end;
font={'interpreter','none'};

if (nargin>1)&(~all(isreal(x(:))));x=full(x);y=full(y);
if nargin<4;nf=[];else nf=parm;end;if isempty(nf);nf=1;end; % nb figures (1 par defaut sinon 4) 
if nargin<3;parm=[];else parm=a;end;if isempty(parm);parm=0;end;
labx=inputname(1);laby=inputname(2);

if nf==1;retsubplot(2,2,1);else figure;end;
retcolor(real(x),imag(x),real(y),parm);xlabel(['real(',labx,')'],font{:});ylabel(['imag(',labx,')'],font{:});zlabel(['real(',laby,')'],font{:}); title([labx,' --> ',laby,'     real',laby],font{:});
if nf==1;retsubplot(2,2,2);else figure;end;
retcolor(real(x),imag(x),imag(y),parm);xlabel(['real(',labx,')'],font{:});ylabel(['imag(',labx,')'],font{:});zlabel(['imag(',laby,')',font{:}]);title([labx,' --> ',laby,'     imag',laby],font{:});
if isreal(parm(1))|(length(parm)>1);return;end;
if nf==1;retsubplot(2,2,3);else figure;end;
retcolor(real(x),real(y),imag(y),parm);xlabel(['real(',labx,')'],font{:});ylabel(['real(',laby,')'],font{:});zlabel(['imag(',laby,')'],font{:});title(['FONCTION INVERSE   ',laby,' --> ',labx],font{:});
if nf==1;retsubplot(2,2,4);else figure;end;
retcolor(imag(x),real(y),imag(y),parm);xlabel(['imax(',labx,')'],font{:});ylabel(['real(',laby,')'],font{:});zlabel(['imag(',laby,')'],font{:});title(['FONCTION INVERSE   ',laby,' --> ',labx],font{:});
return;
end;  % fin fonctions complexes --------------------------------------------------------------------------------------------------

if nargin>=3;labx=inputname(1);laby=inputname(2);laba=inputname(3);else;labx='';laby='';laba='';end;
if nargin<4;parm=[];end;if isempty(parm);parm=0;end;parmm=parm;parm=parm(1);

if nargin<3;%  pcolor(a) 
if nargin<2;parm=0;else parm=y;end;parmm=parm;parm=parm(1);
a=squeeze(full(x));x=[1:size(a,2)];y=[1:size(a,1)];
end;

% pcolor(x,y,a)
x=squeeze(full(x));y=squeeze(full(y));% si a x ou y ont 1 terme:on reformate a (au cas ou size(a)=[1,100,1] ,length(y)=100 ,length(x)=1 ..)
if length(x)==1|length(y)==1;a=reshape(full(a),length(y),length(x));else;a=squeeze(full(a));end;
if imag(parm)==0;  % remise en forme de x y a
  if (size(x,1)==1)|(size(x,2)==1);x=x(:).';end;if (size(y,1)==1)|(size(y,2)==1);y=y(:).';end;    
if (size(x,1)>1)&(size(x,2)>1);x=x(1,:);end;if (size(y,1)>1)&(size(y,2)>1);y=y(:,1).';end;
if x(end)>x(1);[prv,iii]=sort(x);else;[prv,iii]=sort(-x);end;
if y(end)>y(1);[prv,jjj]=sort(y);else;[prv,jjj]=sort(-y);end;
x=x(iii);y=y(jjj);a=a(jjj,iii);
a=[a,a(:,end)];a=[a;a(end,:)];
if length(x)>1;x0=x(1);x1=x(end);x=(x(2:end)+x(1:end-1))/2;x=[2*x0-x(1),x,2*x1-x(end)];else x=[x(1)-.5,x(1)+.5];end;
if length(y)>1;y0=y(1);y1=y(end);y=(y(2:end)+y(1:end-1))/2;y=[2*y0-y(1),y,2*y1-y(end)];else y=[y(1)-.5,y(1)+.5];end;
end;

if imag(parm)==0;pcolor(x,y,a);xlabel(labx,font{:});ylabel(laby,font{:});title(laba,font{:});
else;
if imag(parm)<0;% <---------------contours
if length(parmm)==1;[c,h]=contourf(x,y,a,ceil(abs(parm)));    
else;[c,h]=contourf(x,y,a,parmm(2:end));end;if imag(parm)==fix(imag(parm));clabel(c,h,'fontsize',5);end;colormap(1-gray/2);

else;%            <---------------traces
    
if imag(parm)==2;mesh(x,y,a);colormap('default');end;

if imag(parm)==3; %  stereo --------------------------------------------------------------------------------
r=[1,0,0];v=[0,1,0];   % couleurs des lunettes 
if length(x(:))~=length(a(:));[x,y]=meshgrid(x,y);end;
fig=figure;hold on;plot3(x,y,a);b=get(gca);close(fig);
mx=min(x(:));my=min(y(:));mz=min(a(:));
Mx=max(x(:));My=max(y(:));Mz=max(a(:));
xx=(x-mx)/(Mx-mx);b.XTick=(b.XTick-mx)/(Mx-mx);
yy=(y-my)/(My-my);b.YTick=(b.YTick-my)/(My-my);
zz=(a-mz)/(Mz-mz);b.ZTick=(b.ZTick-mz)/(Mz-mz);
fig=figure(gcf);set(fig,'color',[1,1,1]);hold on;
if (size(zz,1)~=1)&(size(zz,2)~=1);% lignes
h1=mesh(xx,yy,zz,'EdgeAlpha',.5,'EdgeColor',r);hidden off;
h2=mesh(xx,yy,zz,'EdgeAlpha',.5,'EdgeColor',v);hidden off;
else;          % points
h1=plot3(xx,yy,zz,'LineStyle','none','MarkerSize',5,'Marker','.','Color',r);hidden off;
h2=plot3(xx,yy,zz,'LineStyle','none','MarkerSize',5,'Marker','.','Color',v);hidden off;
end;
set(gca,'XTick',b.XTick,'XTickLabel',b.XTickLabel,'XTickLabelMode','manual','XTickMode','manual','XLimMode','manual',...
'YTick',b.YTick,'YTickLabel',b.YTickLabel,'YTickLabelMode','manual','YTickMode','manual','YLimMode','manual',...
'ZTick',b.ZTick,'ZTickLabel',b.ZTickLabel,'ZTickLabelMode','manual','ZTickMode','manual','ZLimMode','manual');
rotate(h2,[0,0,1],-5);
set(gca,'DataAspectRatio',[1 1 1]);view([45,10]);
end;% fin stereo ----------------------------------------------------------------------------------------------

end; %                         <---------------
if abs(parm)==1;colormap(1-gray);else colormap('default');end;
xlabel(labx,font{:});ylabel(laby,font{:});title(laba,font{:});
return;
end;

persistent gcast posc;if isempty(gcast);gcast=nan;end
shading flat;
if parm>=0;
if gca~=gcast;
pos=get(gca,'position');posc=pos;pos(3)=.90*pos(3);posc(1)=posc(1)+.95*posc(3);posc(3)=.05*posc(3);posc(4)=.95*posc(4);
set(gca,'position',pos);
gcast=gca;
end;
%set(gca,'position',pos,'DataAspectRatioMode','manual');
cx=caxis;if (nargin>4)&(pparm==1);cx(1)=0;end;caxis(cx);
c=colorbar;set(c,'position',posc);
t=get(c,'title');set(c,'fontsize',7);
t=get(c,'xlabel');set(t,'String',num2str(cx,1),'fontsize',6);
%t=get(c,'xlabel');set(t,'String',num2str([min(min(a)),max(max(a))],1),'fontsize',6);
end;
if abs(parm)==1;colormap(1-gray);else colormap('default');end

