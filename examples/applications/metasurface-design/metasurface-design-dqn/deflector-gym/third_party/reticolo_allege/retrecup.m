%recuperation des données d'un graphique pcolor
%apres avoir cliquer dessus
%met les données dans XX pour les abscisses
%dans YY pour les ordonnées
%dans CC pour les couleurs (données de pcolor)
%écrit par P VELHA (c'est moi le fautif si ça marche pas)
mdex=get(gca);%recupere les propriétés du graphique en cours
mdex2=get(mdex(1).Children);%récupère les données afférantes au données
XX=mdex2(1).XData;%récupère des données des abscisses
YY=mdex2(1).YData;%récupère des données des ordonnées
CC=mdex2(1).CData;%récupère des données de couleurs, ce qu'on veut en pcolor