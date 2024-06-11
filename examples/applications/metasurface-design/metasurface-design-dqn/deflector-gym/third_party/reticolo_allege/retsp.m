function ss=retsp(s,m,k);
%function s=retsp(s,m,k);
% s =matrice (s) puissance m  (m entier relatif) k:methode de produit des matrices s
%
% remarque: s peut avoir ete stocké sur fichier par retio(s,1) mais au retour s ne l'est pas
%  sauf si m=1 et que s est deja sur fichier

if m==1;ss=s;return;end;
s=retio(s);

	if size(s,2)==2;% matrices T ou S du 0D
	ss=s;ss{1,1}(:,:)=1;ss{2,2}(:,:)=1;ss{1,2}(:,:)=0;ss{2,1}(:,:)=0;
	if m<0;% inversion
	prv=s{1,1}.*s{2,2}-s{1,2}.*s{2,1};prv(prv==0)=1.e-50;% pour eviter les divisions par 0 dans le cas des metaux infiniment conducteurs
	prv=1./prv;
	s{1,1}=s{1,1}.*prv;
	s{2,2}=s{2,2}.*prv;
	s{1,2}=-s{1,2}.*prv;
	s{2,1}=-s{2,1}.*prv;
	[s{1,1},s{2,2}]=deal(s{2,2},s{1,1});
	m=-m;
	end;
   
else;% matrices S ou G 

if size(s,1)<6;n=s{2};ss=rets1(n,1);else;n=s{3};ss=rets1(n,0);end;if m==0;return;end; %unite matrices s ou g
if m<0;s{1}=inv(s{1});m=-m;end;   
end;

% sss=s;
% if nargin<3;
% while(m>=1);if(mod(m,2)==1) ss=retss(sss,ss);end;if(m>0) sss=retss(sss,sss);m=floor(m/2);end;end;   
% else;
% while(m>=1);if(mod(m,2)==1) ss=retss(sss,ss,k);end;if(m>0) sss=retss(sss,sss,k);m=floor(m/2);end;end;   
% end;sss=s;

if nargin<3;
%while(m>=1);if(mod(m,2)==1);ss=retss(s,ss);end;if(m>0);s=retss(s,s);m=floor(m/2);end;end;   
while(m>=1);if (mod(m,2)==1);if isempty(ss);ss=s;else;ss=retss(s,ss);end;end;m=floor(m/2);if m>0 ;s=retss(s,s);end;end;   
else;
%while(m>=1);if(mod(m,2)==1);ss=retss(s,ss,k);end;if(m>0);s=retss(s,s,k);m=floor(m/2);end;end;   
while(m>=1);if (mod(m,2)==1);if isempty(ss);ss=s;else;ss=retss(s,ss,k);end;end;m=floor(m/2);if m>0 ;s=retss(s,s,k);end;end;   
end;