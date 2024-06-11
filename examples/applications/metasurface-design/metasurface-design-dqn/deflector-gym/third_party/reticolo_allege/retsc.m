function uv=retsc(ss,s,inc,m);
% function uv=retsc(Sh,Sb,Inc,m);
%      |
%     Ih    /\
%     \/    Dh               |       Dh    |            | uv(1:m) |
% ---------- |               |             |   = Sh *   |         |
%     Sh                     | uv(m+1:2*m) |            |  Ih     |
% **********  uv (1:2*m)
%              vecteur       | uv(1:m) |            |    Ib       |
%     Sb                     |         |   = Sb *   |             |
%                            |  Db     |            | uv(m+1:2*m) |
% ---------- |
%     /\    Db               Inc=[Ib,Ih] vecteur champ incident 
%     Ib    \/ 
%      |
%    
%    
% calcul du champ uv au bas de la matrice Sh (et au haut de la matrice Sb )
%   La matrice totale est S=Sh*Sb
%  Sh(nn3+nn4,nn1+nn2)  Sb(n3+n4,n1+n2)  Inc(n1+nn2)   uv a pour longueur 2*m  
%  les matrices S pouvant etre de n'importe quel type (champ<--> mode mode <--> champ ...)
%  uv peut etre soit un champ soit des modes
%  Marche aussi en G
% See also RETCHAMP
test=1;try;
inc=inc(:);
[ss,s]=retcrs(ss,s);  %croissance eventuelle des matrices 

if size(s,1)<6;  % matrices S
n1=s{2};n2=size(s{1},2)-n1;n3=s{3};n4=size(s{1},1)-n3;
nn1=ss{2};nn2=size(ss{1},2)-nn1;nn3=ss{3};nn4=size(ss{1},1)-nn3;
%uv=[[eye(n3),-s{1}(1:n3,n1+1:n1+n2)];[-ss{1}(nn3+1:nn3+nn4,1:nn1),eye(n2)]]\[s{1}(1:n3,1:n1)*inc(1:n1,1);ss{1}(nn3+1:nn3+nn4,nn1+1:nn1+nn2)*inc(n1+1:n1+nn2,1)];
uv=[s{1}(1:n3,1:n1)*inc(1:n1,1);ss{1}(nn3+1:nn3+nn4,nn1+1:nn1+nn2)*inc(n1+1:n1+nn2,1)];
s=s{1}(1:n3,n1+1:n1+n2);ss=ss{1}(nn3+1:nn3+nn4,1:nn1);

% [rcond(eye(n3)-s*ss),rcond(eye(n2)-ss*s),rcond([[eye(n3),-s];[-ss,eye(n2)]])]
% [prv,kk]=max([rcond(eye(n3)-s*ss),rcond(eye(n2)-ss*s),rcond([[eye(n3),-s];[-ss,eye(n2)]])]);
kk=4;
switch kk;
case 1;% % version instable ?
prv=uv(n3+1:n3+n2);
uv(1:n3)=(eye(n3)-s*ss)\(uv(1:n3)+s*prv);clear s;
uv(n3+1:n3+n2)=prv+ss*uv(1:n3);
case 2;% % version instable ?
prv=uv(1:n3);
uv(n3+1:n3+n2)=(eye(n2)-ss*s)\(uv(n3+1:n3+n2)+ss*prv);clear ss;
uv(1:n3)=prv+s*uv(n3+1:n3+n2);
case 3;% % version stable ?
s=[eye(n3),-s];ss=[-ss,eye(n2)];s=[s;ss];clear ss;uv=s\uv;

case 4;% % version stable ?
s=[eye(n3),-s];ss=[-ss,eye(n2)];s=[s;ss];clear ss;
prv=max(abs(s),[],1);s=s*retdiag(1./prv);
pprv=max(abs(s),[],2);s=retdiag(1./pprv)*s;
uv=(1./prv).'.*(s\((1./pprv).*uv));
end;% kk

uv=[uv(1:m);uv(n3+1:n3+m)];

else;            % matrices G
n1=s{3};n2=s{4};n3=s{5};n4=s{6};n=size(s{1},1);
nn1=ss{3};nn2=ss{4};nn3=ss{5};nn4=ss{6};nn=size(ss{1},1);
%  rcond(([[s{2},-s{1}(:,n1+1:n1+n2),zeros(n,nn3)];[ss{1},zeros(nn,n2),-ss{2}(:,1:nn3)]]))
uv=([[s{2},-s{1}(:,n1+1:n1+n2),zeros(n,nn3)];[ss{1},zeros(nn,n2),-ss{2}(:,1:nn3)]])\...
    (([[s{1}(:,1:n1),zeros(n,nn4)];[zeros(nn,n1),ss{2}(:,nn3+1:nn3+nn4)]])*inc);
uv=[uv(1:m);uv(n3+1:n3+m)];

end;

catch;test=0;end;if test==0;texte=lasterror;error([texte.message,' Dimension de inc dans retchamp incompatible avec la troncature de sh et sb ?']);end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
