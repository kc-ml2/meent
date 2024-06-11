function e=retener(s,haut,bas);
% function e=retener(s,haut,bas);
% test de conservation de l'energie
% haut,bas calcules par retb ou numeros des ordres diffractifs (tenant compte d'eventuelles troncatures de S)
% le test est fait colonne par colonne et doit donner 0 pour toutes les colonnes
%

if isempty(haut)|isempty(bas) e=0;return;end;
if size(haut,2)>1;haut=haut(:,1);end;if size(bas,2)>1;bas=bas(:,1);end;
n1=s{2};n2=s{3};
e=sum(abs(s{1}([haut;bas+n2],[bas;haut+n1])).^2)-1;