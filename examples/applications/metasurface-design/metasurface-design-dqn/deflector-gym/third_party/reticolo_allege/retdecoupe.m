function [ww,whaut,wbas]=retdecoupe(w,z1,z2);
% function [ww,whaut,wbas]=retdecoupe(w,z1,z2);
%  decoupe dans un maillage de tronçon de la partie comprise entre z1 et z2
%  z1>=0 z2>=0 si z2> hauteur totale z2=hauteur totale
% whaut,wbas;tronçons des textures du haut (du bas) avec une epaisseur nulle (pour les sources)
% See also RETW1 RETAUTOMATIQUE RETROTATION
z1=max(z1,0);
n=length(w);
h=zeros(1,n+1);for ii=n:-1:1;h(n-ii+2)=h(n-ii+1)+w{ii}{1};end;z2=min(z2,h(end));
hh=retelimine([h,z1,z2]);
hh=hh((hh>=z1)&(hh<=z2));
centres=(hh(1:end-1)+hh(2:end))/2;nn=length(centres);
ww=cell(1,nn);
for ii=1:nn;
jj=find((h(1:end-1)<centres(ii))&(h(2:end)>centres(ii)));
if ~isempty(jj);
ww{nn-ii+1}={hh(ii+1)-hh(ii),w{n-jj+1}{2}};
end;
end;
if isempty(ww);ww=w(end);ww{1}{1}=0;end;
if nargout>1;whaut=ww(1);whaut{1}{1}=0;end;
if nargout>2;wbas=ww(end);wbas{1}{1}=0;end;

