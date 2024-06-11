function hand=retsubplot(kx,ky,k,varargin);
% % 	retsubplot(kx,ky,k,options)
% % 	si kx,ky,k sont reels, idem a subplot mais option align en matlab 7
% %
% % 	sinon:
% % 	on partage les axes sans perte d'espace.C'est ensuite a l'utilisateur de creer des marges par appel a 
% % 	retsubplot(marges),(ou retsubplot(handel,marges))
% % 	marge tableau de 4 elements: [ gauche,bas,droit,haut] en unites reduites
% %        (1--> toute la hauteur ou la largeur de la page)
% %   ------------------------------
% %   |   marges     |          m   |
% %   |              |          a   |
% %   |   *******    | *******  r   |
% %   |   *******    | *******  g   |
% %   |   *PLOT**    | *******  e   |
% %   |   *******    | *******  s   |
% %   |              |              |
% %  --------------------------------
% %    <dim initiale>
% % 
% 	%% EXEMPLES
% 	figure;m=6;x=linspace(-2,2,100);
% 	marges=[.08,0,.02,.01];   % marges ajoutees a chaque figure:
% 	offset=linspace(-0.05,.08,m+1);% pour faire 'glisser' progressivement les marges et liberer de la place en bas pour l'axe et xlabel
% 	for ii=1:m;retsubplot(m,1+i,ii,'fontsize',8);
% 	plot(x,sin(ii*x));ylabel('Y');grid;
% 	if ii>1;yt=get(gca,'YTick');set(gca,'YTick',yt(1:end-1));else;title('exemple de courbes');end;
% 	if ii<m;xt=get(gca,'XTick');set(gca,'XTickLabel',cell(size(xt)));else;xlabel('X');end;
% 	retsubplot(marges+[0,offset(ii+1),0,-offset(ii)]);
% 	end
%   figure;for ii=1:m;retsubplot(m,1,ii);plot(x,sin(ii*x));xlabel('X');ylabel('Y');if ii==2;title({'exemple','de','retsubplot(m,1,ii)'});end;grid;end;
%   figure;for ii=1:m;retsubplot(m,1-i,ii);plot(x,sin(ii*x));xlabel('X');ylabel('Y');if ii==2;title({'exemple','de','retsubplot(m,1-i,ii)'});end;grid;end;
%   figure;for ii=1:m;retsubplot(m,1+i,ii);plot(x,sin(ii*x));xlabel('X');ylabel('Y');if ii==2;title({'exemple','de','retsubplot(m,1+i,ii)'});end;grid;end;


switch(nargin);case 0;retmarges;return;case 1;retmarges(kx);return;case 2;retmarges(kx,ky);return;end;
if isreal(kx)&isreal(ky)&isreal(k);     % <---------- version matlab
vers=retversion;
if vers>6;if nargin==3;varargin=[varargin,{'align'}];end;
else;varargin={};end;%varargin(1:3);end;     
if nargout==1;hand=subplot(kx,ky,k,varargin{:});else;subplot(kx,ky,k,varargin{:});end;
else;%<---------------- version 'reticolo'
type=(imag(kx)>0)|(imag(ky)>0)|(imag(k)>0);
kx=real(kx);ky=real(ky);k=real(k);
[iy,ix]=ind2sub([ky,kx],k);
if type;axes('Position',[(iy-1)/ky,1-ix/kx,1/ky,1/kx],varargin{:});
else;axes('OuterPosition',[(iy-1)/ky,1-ix/kx,1/ky,1/kx],varargin{:});end;

end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function retmarges(h,marges);
if nargin<1;h=gca;end;
if nargin<2;marges=[0,0,0,0];end;if isempty(marges);marges=[0,0,0,0];end;
if length(h)>1;marges=h;h=gca;end;

if length(marges)==2;marges=[marges(1)/2,marges(2)/2,marges(1)/2,marges(2)/2];end;
if length(marges)==1;marges=[marges/2,marges/2,marges/2,marges/2];end;
Position=get(h,'Position');
set(h,'Position',[Position(1)+marges(1),Position(2)+marges(2),Position(3)-marges(1)-marges(3),Position(4)-marges(2)-marges(4)]);


