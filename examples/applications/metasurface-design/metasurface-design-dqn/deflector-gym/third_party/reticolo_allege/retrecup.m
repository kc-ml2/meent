%recuperation des donn�es d'un graphique pcolor
%apres avoir cliquer dessus
%met les donn�es dans XX pour les abscisses
%dans YY pour les ordonn�es
%dans CC pour les couleurs (donn�es de pcolor)
%�crit par P VELHA (c'est moi le fautif si �a marche pas)
mdex=get(gca);%recupere les propri�t�s du graphique en cours
mdex2=get(mdex(1).Children);%r�cup�re les donn�es aff�rantes au donn�es
XX=mdex2(1).XData;%r�cup�re des donn�es des abscisses
YY=mdex2(1).YData;%r�cup�re des donn�es des ordonn�es
CC=mdex2(1).CData;%r�cup�re des donn�es de couleurs, ce qu'on veut en pcolor