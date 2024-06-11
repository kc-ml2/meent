

function retmarges(h,marges);
% creation de marges autour d'un objet 'axes' defini par axes('OuterPosition',[*,*,*,*])
% de maniere a ne par masquer titre ,axes ,legendes,.. 
%  Apres creation de ces elements : 
%    retmarges
%   retmarges(marges) marges vecteur de longueur 2 (marges laterales,marges verticales)  ou 4 (gauche,bas,droit,haut)
%   retmarges(h,marges)  h handel de l'objet 'axes' 
%  l'unite de marges est la hauteur et la largeur du graphique total
%% EXEMPLES
% figure;
% for ii=0:.2:.8;
% axes('OuterPosition',[0,ii,1,.2],'fontsize',8);
% plot(rand(1,500));title(['courbe numero: ',num2str(5*(ii+.2))]);retmarges([-.02,-.05]);
% end;
% 
% figure('color','w');
% axes('OuterPosition',[0,0,.75,1],'fontsize',12);plot(rand(1,500));
% title({'exemple','de','titre'});legend('courbe numero: 1');xlabel('x');ylabel('y');retmarges;
% axes('OuterPosition',[0.75,0.25,.25,.75],'fontsize',5);plot(rand(1,500));retmarges([.1,.1]);
% axes('OuterPosition',[0.75,0,.25,.25],'fontsize',5);plot(rand(1,500));axis off;retmarges;





if nargin<1;h=gca;end;
if nargin<2;marges=[0,0,0,0];end;if isempty(marges);marges=[0,0,0,0];end;
if length(h)>1;marges=h;h=gca;end;

if length(marges)==2;marges=[marges(1)/2,marges(2)/2,marges(1)/2,marges(2)/2];end;
OuterPosition=get(h,'OuterPosition');TightInset=get(h,'TightInset')+marges;
set(h,'Position',[OuterPosition(1)+TightInset(1),OuterPosition(2)+TightInset(2),OuterPosition(3)-TightInset(1)-TightInset(3),OuterPosition(4)-TightInset(2)-TightInset(4)]);


