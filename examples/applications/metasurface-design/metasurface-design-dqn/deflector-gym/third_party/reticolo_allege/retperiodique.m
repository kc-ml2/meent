function [s,ss]=retperiodique(s,a) 
% [Champ,Inc]=retperiodique(s,a) 
% multiplication par un facteur a (periodisation ou pseudo périodisation)
% s matrice S entre composantes de Fourier, contenant des sources
% On veut exprimer que de bas en haut le champ est multiplié par a 
% 
% Si les sources sont un vecteur colonne Inc_sources, les composantes incidentes [Ebas;Hhaut;Inc_sources]
% sont Inc*Inc_sources, et le champ sur la source Champ*Inc_sources
% Peut être utilisé dans retchamp avec: 
% 	sh=rettronc(rets1(init),0,[],1) 
% 	sb=rettronc(rets1(init),0,[],-1)
% 	inc=Inc*Inc_sources
% Attention dans sh et sb,il faut garder toutes les composantes incidentes, mais les sources peuvent être tronquées
retio(s);
n=s{2};s=s{1};nsi=size(s,2)-2*n;nsd=size(s,1)-2*n;
ss=[[a*eye(n)-s(1:n,1:n),-s(1:n,n+1:2*n)];[-s(n+nsd+1:2*n+nsd,1:n),(1/a)*eye(n)-s(n+nsd+1:2*n+nsd,n+1:2*n)]]\[s(1:n,2*n+1:2*n+nsi);s(n+nsd+1:2*n+nsd,2*n+1:2*n+nsi)];
s=s(n+1:n+nsd,1:2*n)*ss+s(n+1:n+nsd,2*n+1:2*n+nsi);
ss=[ss;eye(nsi)];% modif 9 2009
