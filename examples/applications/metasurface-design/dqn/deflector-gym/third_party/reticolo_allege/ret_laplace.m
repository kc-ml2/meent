function p=ret_laplace(t,p,nfix,tol);	
% p=ret_laplace(t,p,nfix,tol);	
% tol pour determiner les elements à traiter
% nfix de 1 à nfix on ne change pas les points

p_store=p;
prv=ret_vol_maillage(t,p,struct('cal_Rs',1));
t_mauvais=t(prv.Rs<tol,:);

if size(p,2)==3; % 2D ***********************
prv=retelimine(t_mauvais);
T=t(ismember(t(:,1),prv)|ismember(t(:,2),prv)|ismember(t(:,3),prv)|ismember(t(:,4),prv),:);% tous les voisins des mauvais
prv=ret_vol_maillage(t_mauvais,p,struct('cal_Sf',1));
[prv,kk]=max([double(t_mauvais(:,1)>nfix).*prv.Sf(:,1),double(t_mauvais(:,2)>nfix).*prv.Sf(:,2),double(t_mauvais(:,3)>nfix).*prv.Sf(:,3),double(t_mauvais(:,4)>nfix).*prv.Sf(:,4)],[],2);	
f=find(prv>0);kk=kk(f);t_mauvais=t_mauvais(f,:);% on elimine les points fixes
for ii=1:length(kk);ttt=t_mauvais(ii,kk(ii));
% 	% tt=setdiff(retelimine(T((T(:,1)==ttt)|(T(:,2)==ttt)|(T(:,3)==ttt)|(T(:,4)==ttt),:)),ttt);
% 	% p(ttt,:)=mean(p(tt,:),1);
% 	kkk=1+mod(kk(ii)+[-1:2],4);
% 	V=retvectoriel(p(t_mauvais(ii,kkk(3)),:)-p(t_mauvais(ii,kkk(2)),:),p(t_mauvais(ii,kkk(4)),:)-p(t_mauvais(ii,kkk(2)),:));
% 	%Sbase=sqrt(sum(v.^2,2))/2;v=retdiag(.5./Sbase)*v;
% 	V=(1/sqrt(sum(V.^2,2)))*V;
% 	lmax=sqrt(max([sum((p(t_mauvais(ii,kkk(3)),:)-p(t_mauvais(ii,kkk(2)),:)).^2),sum((p(t_mauvais(ii,kkk(3)),:)-p(t_mauvais(ii,kkk(4)),:)).^2),sum((p(t_mauvais(ii,kkk(4)),:)-p(t_mauvais(ii,kkk(2)),:)).^2)],[],2));
% 	p(ttt,:)=p(ttt,:)+.5*((p(t_mauvais(ii,kkk(2)),:)+p(t_mauvais(ii,kkk(3)),:)+p(t_mauvais(ii,kkk(4)),:))/3+(lmax*sqrt(2/3))*V-p(ttt,:));
TT=T(retelimine(find((T(:,1)==ttt)|(T(:,2)==ttt)|(T(:,3)==ttt)|(T(:,4)==ttt))),:);% tetraedres adjacents

 %figure;tetramesh(TT,p,'facealpha',.1);hold on;tetramesh(t_mauvais(ii,:),p);plot3(p(ttt,1),p(ttt,2),p(ttt,3),'or','markersize',8);
[p(ttt,:),FVAL]=fminsearch(@err,p(ttt,:), optimset('TolFun',1.e-2,'TolX',1.e-2),ttt,TT,p);
end;	

	
else;             % 1D  ***********************
prv=retelimine(t_mauvais);
T=t(ismember(t(:,1),prv)|ismember(t(:,2),prv)|ismember(t(:,3),prv),:);% tous les voisins des mauvais
prv=ret_vol_maillage(t_mauvais,p,struct('cal_La',1));
[prv,kk]=max([double(t_mauvais(:,1)>nfix).*prv.La(:,1),double(t_mauvais(:,2)>nfix).*prv.La(:,2),double(t_mauvais(:,3)>nfix).*prv.La(:,3)],[],2);
f=find(prv>0);kk=kk(f);t_mauvais=t_mauvais(f,:);% on elimine les points fixes

for ii=1:length(kk);ttt=t_mauvais(ii,kk(ii));
tt=setdiff(retelimine(T((T(:,1)==ttt)|(T(:,2)==ttt)|(T(:,3)==ttt),:)),ttt);
p(ttt,:)=mean(p(tt(:),:),1);
end;	

end;            % 1D 2D  ***********************
% cas d'echec
ff=1;while ~isempty(ff);
prv=ret_vol_maillage(t,p,struct('cal_V',1));
ff=find(prv.V<0);
prv=retelimine(t(ff,:));
p(prv,:)=p_store(prv,:);
end;

function er=err(P,ttt,t,p);
p(ttt,:)=P;
prv=ret_vol_maillage(t,p,struct('cal_Rv',1));
er=1/min(prv.Rv);
if er<0;er=1.e6;end;