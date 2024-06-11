function ef=retreseau(init,s,betb,cb,anglb,incb,difb,beth,ch,anglh,inch,difh);
%  function ef=retreseau(init,s,betb,cb,anglb,incb,difb,beth,ch,anglh,inch,difh);
%  s matrice S ou G totale sh et sb ayant ete construites et tronquees comme suit :
%
% [sh,haut,beth,ch,anglh]=retb(init,ah,1.e-6);
%   sh=rettronc(sh,haut(inch,1),haut(difh,1),1);% inch difh numeros choisis dans les lignes de haut
% [sb,bas,betb,cb,anglb]=retb(init,ab,-1.e-6);
%   sb=rettronc(sb,bas(incb,1),bas(difb,1),-1);%  incb difb numeros choisis dans les lignes de bas
% 
% si inch difh  incb ou difb =0: on prend tous les ordres. Il faut alors remplacer aussi haut(inch,1) par 0 ,etc...
% par exemple: inch=0; difh=0;sh=rettronc(sh,0,0,1);
%
% la structure ef est la 'carte d'identitee'  des ondes planes incidentes et diffractees  propagatives 
%  et fournit en outre les amplitudes diffractees par une onde incidente 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                 inc:  [1x1 struct]            ordres incidents                                 %%%%%%%%%%%%%%%%%
%       ef.       dif: [1x1 struct]           ordres diffractes                                  %     2 D       %
%                amplitude: [46x4 double]    amplitudes diffractees                              %%%%%%%%%%%%%%%%ù
%
%
% 
%           teta: [6.6478 6.6478 10.0000 10.0000]       angles teta(ligne)
%            delta: [20 20 20.0000 20.0000]   angles delta(ligne)
%              psi: [90 0 90 0]   angles psi(ligne)
%             beta: [2x4 double]  beta composantes x y du vecteur d'onde
%            ordre: [2x4 double]  ordres de diffraction
%           zordre: [1x4 double]  ordres de diffraction en complexe(commode a utiliser pour chercher un ordre donne)
%                E: [2x4 double]  composantes de E dans le repere u v (ligne 1:composante sur u   ligne 2:composante sur v)
%                H: [2x4 double]  composantes de H dans le repere u v
%            champ: [6x4 double]  [Ex;Ey;Ez;Hx;Hy;Hz] a l'origine
%             sens: [1 1 -1 -1]   sens de propagation 1:vers le haut  -1 vers le bas
%                h: [3 4]  numeros des incidents du haut
%   ef.inc.      b: [1 2]  numeros des incidents du bas
%               TE: [1 3]  numeros des incidents TE
%               TM: [2 4]  numeros des incidents TM
%              TEh: 3  numeros des incidents  du haut TE    \
%              TMh: 4  numeros des incidents du haut TM      |        en general un element unique 
%              TEb: 1  numeros des incidents  du bas TE      |  mais certains peuvent etre absents a cause des symetries
%              TMb: 2  numeros des incidents  du bas TM     /
%
%
%
%
%            teta: [1x46 double]    angles teta(ligne)
%            delta: [1x46 double]   angles delta(ligne)
%              psi: [1x46 double]   angles psi(ligne)
%             beta: [2x46 double]   beta composantes x y du vecteur d'onde
%            ordre: [2x46 double]   ordres de diffraction
%           zordre: [1x46 double]   ordres de diffraction en complexe(commode a utiliser pour chercher un ordre donne)
%                E: [2x46 double]   composantes de E dans le repere u v
%                H: [2x46 double]   composantes de H dans le repere u v
%            champ: [6x46 double] [Ex;Ey;Ez;Hx;Hy;Hz] à l'origine
%             sens: [1x46 double]   sens de propagation 1:vers le haut  -1 vers le bas
%                h: [1 2 3 4 5 6 7 8 9 10 11 12 13 14]  numeros des diffractes en haut
%  ef.dif.       b: [15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46]
%                                     numeros des diffractes en bas
%               TE: [1 2 3 4 5 6 7 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]        numeros des diffractes TE
%               TM: [8 9 10 11 12 13 14 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46]  numeros des diffractes TM
%              TEh: [1 2 3 4 5 6 7]  numeros des diffractes  en haut TE
%              TMh: [8 9 10 11 12 13 14]  numeros des diffractes en haut TM
%              TEb: [15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]  numeros des diffractes  en bas TE
%              TMb: [31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46]  numeros des diffractes  en bas TM
%
%     l' amplitude de ces ondes planes est normalisee proportionnellement au flux d'energie à travers le plan x y
%       (les champs E et H d'une onde normale a ce plan dans un milieu d'indice 1 etant E=1  H=1 )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                 inc:        ordres incidents                                    %%%%%%%%%%%%%%%%%
%       ef.       dif:        ordres diffractes                                   %     1 D       %
%                amplitude:   amplitudes diffractees                              %%%%%%%%%%%%%%%%ù
% 
%             teta:   angles teta(ligne)
%             beta:[2x4 double]   beta composantes x  du vecteur d'onde
% ef.inc.    ordre:  ordre de diffraction
%                E:  composante de E sur z
%                H:  composante de H sur v
%            champ: [Ez;Hx;Hy] a l'origine
%             sens:  sens de propagation 1:vers le haut  -1 vers le bas
%                h:  numero des incident du haut
%                b:  numero des incident du bas
%
%             teta:   angles teta(ligne)
%             beta: beta composantes x du vecteur d'onde
% ef.dif.    ordre:  ordres de diffraction
%                E:  composante de E sur z
%                H:  composante de H sur v
%            champ:  [Ez;Hx;Hy] a l'origine
%             sens:  sens de propagation 1:vers le haut  -1 vers le bas
%                h:  numeros des diffractes en haut
%                b:  numeros des diffractes en bas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                        | numero de l'onde incidente  |
%                     n  ------------------------------
%                     u  |      |      |       |       |
%                     m  |      |      |       |       |   
%                     .  |      |      |       |       |
%                     o  |      |      |       |       | 
%                     n  |      |      |       |       | 
%    ef.amplitude     d  |      |      |       |       | 
%                     e  |      |      |       |       |  
%                        |      |      |       |       |
%                     d  |      |      |       |       | 
%                     i  |      |      |       |       | 
%                     f  |      |      |       |       | 
%                     .  -------------------------------
%           matrice formee par les colonnes amplitudes diffractees par les incidents(lignes) 
%
%      par exemple  en 2 D:
%      les efficacitees diffractees par l'onde TE incidente du haut sur les ondes TM en bas
%      forment le vecteur colonne:   abs(  ef.amplitude(ef.dif.TMb,ef.inc.TEh) ).^2  
%
%      la somme des efficacitees diffractees par l'onde TM incidente du bas dans tout l'espace est
%             sum(  abs(ef.amplitude(:,ef.inc.TMb)).^2 )
%
%     certains tableaux peuvent etre vides
%
% See also:RETB,RETTRONC

sym=init{end}.sym;beta0=init{end}.beta0;d=init{end}.d;
if ~isempty(incb);if incb==0;incb=1:size(betb,1);end;end;
if ~isempty(inch);if inch==0;inch=1:size(beth,1);end;end;
if ~isempty(difb);if difb==0;difb=1:size(betb,1);end;end;
if ~isempty(difh);if difh==0;difh=1:size(beth,1);end;end;
    
    

 if init{end}.dim==2;             %%%%%%%%%%%%%
                                  %    2 D    %
                                  %%%%%%%%%%%%%


cm=[cb(:,1:2),zeros(size(cb(:,1))),cb(:,3:4),zeros(size(cb(:,1)))];
cp=[cb(:,1:2),zeros(size(cb(:,1))),-cb(:,3:4),zeros(size(cb(:,1)))];% E->E   H-> -H
poldb={anglb{1}(difb),anglb{2}(difb),anglb{11}(difb),anglb{15}(difb,:),anglb{16}(difb,:),betb(difb,:),cm(difb,:)};% diffractes en bas
%        teta     delta     psim        eem       hhm    beta  champ
polib={anglb{1}(incb),anglb{2}(incb),anglb{3}(incb),anglb{7}(incb,:),anglb{8}(incb,:),betb(incb,:),cp(incb,:)};% incidents en bas
%        teta                  delta            psim               eep                hhp    beta     champ

cm=[ch(:,1:2),zeros(size(ch(:,1))),ch(:,3:4),zeros(size(ch(:,1)))];
cp=[ch(:,1:2),zeros(size(ch(:,1))),-ch(:,3:4),zeros(size(ch(:,1)))];% E->E   H-> -H
polih={anglh{1}(inch),anglh{2}(inch),anglh{11}(inch),anglh{15}(inch,:),anglh{16}(inch,:),beth(inch,:),cm(inch,:)};% incidents en haut
%        teta                  delta            psim               eem                hhm      beta        champ
poldh={anglh{1}(difh),anglh{2}(difh),anglh{3}(difh),anglh{7}(difh,:),anglh{8}(difh,:),beth(difh,:),cp(difh,:)};% diffractes en haut
%        teta     delta     psip      eep       hhp   beta champ

s=reteval(s);


% construction de la structure 'conviviale' ef
 inc=struct('teta',[polib{1}.',polih{1}.'],'delta',[polib{2}.',polih{2}.'],'psi',[polib{3}.',polih{3}.'],...
'beta',[polib{6}.',polih{6}.'],'ordre',[],'zordre',[],'E',[polib{4}.',polih{4}.'],'H',[polib{5}.',polih{5}.'],'champ',[polib{7}.',polih{7}.'],...
'sens',[ones(1,length(polib{1})),-ones(1,length(polih{1}))],'h',[],'b',[],'TE',[],'TM',[],'TEh',[],'TMh',[],'TEb',[],'TMb',[]);
inc.ordre=round([(inc.beta(1,:)-beta0(1))*d(1);(inc.beta(2,:)-beta0(2))*d(2)]/(2*pi));

dif=struct('teta',[poldh{1}.',poldb{1}.'],'delta',[poldh{2}.',poldb{2}.'],'psi',[poldh{3}.',poldb{3}.'],...
'beta',[poldh{6}.',poldb{6}.'],'ordre',[],'zordre',[],'E',[poldh{4}.',poldb{4}.'],'H',[poldh{5}.',poldb{5}.'],'champ',[poldh{7}.',poldb{7}.'],...
'sens',[ones(1,length(poldh{1})),-ones(1,length(poldb{1}))],'h',[],'b',[],'TE',[],'TM',[],'TEh',[],'TMh',[],'TEb',[],'TMb',[]);
dif.ordre=round([(dif.beta(1,:)-beta0(1))*d(1);(dif.beta(2,:)-beta0(2))*d(2)]/(2*pi));

ef=struct('inc',inc,'dif',dif,'amplitude',s);

if isempty(ef.inc.ordre);return;end;
% complement de ef.dif dans le cas de symetries
if ~isempty(sym);
if sym(2)~=0;  % symetrie par rapport a l'axe x
k=find(ef.dif.ordre(2,:)~=0);
ef.amplitude(k,:)=ef.amplitude(k,:)/sqrt(2);    
ef.amplitude=[ef.amplitude;diag(sym(2).*exp(2i*ef.dif.beta(2,k)*sym(4)))* ef.amplitude(k,:)];    
ef.dif.teta=[ef.dif.teta,ef.dif.teta(k)];    
ef.dif.delta=[ef.dif.delta,-ef.dif.delta(k)];    
ef.dif.psi=[ef.dif.psi,-ef.dif.psi(k)];    
ef.dif.beta=[ef.dif.beta,[ef.dif.beta(1,k);-ef.dif.beta(2,k)]];    
ef.dif.ordre=[ef.dif.ordre,[ef.dif.ordre(1,k);-ef.dif.ordre(2,k)]];    
ef.dif.champ(:,k)=ef.dif.champ(:,k)*sqrt(2);   
ef.dif.champ=[ef.dif.champ,diag([1,-1,0,-1,1,0])*ef.dif.champ(:,k)];    
ef.dif.E(:,k)=ef.dif.E(:,k)*sqrt(2);ef.dif.H(:,k)=ef.dif.H(:,k)*sqrt(2);    
ef.dif.E=[ef.dif.E,[ef.dif.E(1,k);-ef.dif.E(2,k)]];    
ef.dif.H=[ef.dif.H,[-ef.dif.H(1,k);ef.dif.H(2,k)]];
ef.dif.sens=[ef.dif.sens,ef.dif.sens(k)];    
end;  

if sym(1)~=0;  % symetrie par rapport a l'axe y
k=find(ef.dif.ordre(1,:)~=0);   
ef.amplitude(k,:)=ef.amplitude(k,:)/sqrt(2);    
ef.amplitude=[ef.amplitude;-diag(sym(1).*exp(2i*ef.dif.beta(1,k)*sym(3)))*ef.amplitude(k,:)];    
ef.dif.teta=[ef.dif.teta,ef.dif.teta(k)];    
ef.dif.delta=[ef.dif.delta,pi-ef.dif.delta(k)];    
ef.dif.psi=[ef.dif.psi,-ef.dif.psi(k)];    
ef.dif.beta=[ef.dif.beta,[-ef.dif.beta(1,k);ef.dif.beta(2,k)]];    
ef.dif.ordre=[ef.dif.ordre,[-ef.dif.ordre(1,k);ef.dif.ordre(2,k)]];    
ef.dif.champ(:,k)=ef.dif.champ(:,k)*sqrt(2);   
ef.dif.champ=[ef.dif.champ,diag([-1,1,0,1,-1,0])*ef.dif.champ(:,k)];    
ef.dif.E(:,k)=ef.dif.E(:,k)*sqrt(2);ef.dif.H(:,k)=ef.dif.H(:,k)*sqrt(2);    
ef.dif.E=[ef.dif.E,[ef.dif.E(1,k);-ef.dif.E(2,k)]];    
ef.dif.H=[ef.dif.H,[-ef.dif.H(1,k);ef.dif.H(2,k)]]; 
ef.dif.sens=[ef.dif.sens,ef.dif.sens(k)];    
end;  
end;

% composantes en z de champ
ef.inc.champ(3,:)=-ef.inc.E(1,:).*sin(ef.inc.teta);
ef.inc.champ(6,:)=-ef.inc.H(1,:).*sin(ef.inc.teta);
ef.dif.champ(3,:)=-ef.dif.E(1,:).*sin(ef.dif.teta);
ef.dif.champ(6,:)=-ef.dif.H(1,:).*sin(ef.dif.teta);

ef.dif.psi=pi/2-mod(pi/2-ef.dif.psi,pi);
% mise en ordre par prioritee: sens ,ordre x,ordre y, TE  ,TM
[prv,nn]=sort(-1.e13*ef.dif.sens+1.e7*ef.dif.ordre(1,:)+10*ef.dif.ordre(2,:)-abs(ef.dif.psi)/(pi*2));
ef.dif.teta=ef.dif.teta(nn);
ef.dif.delta=ef.dif.delta(nn);
ef.dif.psi=ef.dif.psi(nn);
ef.dif.beta=ef.dif.beta(:,nn);
ef.dif.ordre=ef.dif.ordre(:,nn);
ef.dif.E=ef.dif.E(:,nn);
ef.dif.H=ef.dif.H(:,nn);
ef.dif.champ=ef.dif.champ(:,nn);
ef.dif.sens=ef.dif.sens(nn);
ef.amplitude=ef.amplitude(nn,:);


[prv,nn]=sort(-1.e13*ef.inc.sens+1.e7*ef.inc.ordre(1,:)+10*ef.inc.ordre(2,:)-abs(ef.inc.psi)/(pi/2));
ef.inc.teta=ef.inc.teta(nn);
ef.inc.delta=ef.inc.delta(nn);
ef.inc.psi=ef.inc.psi(nn);
ef.inc.beta=ef.inc.beta(:,nn);
ef.inc.ordre=ef.inc.ordre(:,nn);
ef.inc.E=ef.inc.E(:,nn);
ef.inc.H=ef.inc.H(:,nn);
ef.inc.champ=ef.inc.champ(:,nn);
ef.inc.sens=ef.inc.sens(nn);
ef.amplitude=ef.amplitude(:,nn);


% remplissege de 'h' 'b' 'TE'  'TM'  'TMh'  'TMb'  'TEh'  'TEb'
[ef.inc.b,ef.inc.h]=retfind(ef.inc.sens>0);
[ef.inc.TE,ef.inc.TM]=retfind(abs(ef.inc.psi)>pi/4);
[ef.inc.TEh,ef.inc.TMh]=retfind(abs(ef.inc.psi(ef.inc.h))>pi/4);ef.inc.TEh=ef.inc.h(ef.inc.TEh);ef.inc.TMh=ef.inc.h(ef.inc.TMh);
[ef.inc.TEb,ef.inc.TMb]=retfind(abs(ef.inc.psi(ef.inc.b))>pi/4);ef.inc.TEb=ef.inc.b(ef.inc.TEb);ef.inc.TMb=ef.inc.b(ef.inc.TMb);
    
[ef.dif.h,ef.dif.b]=retfind(ef.dif.sens>0);   
[ef.dif.TE,ef.dif.TM]=retfind(abs(ef.dif.psi)>pi/4);
[ef.dif.TEh,ef.dif.TMh]=retfind(abs(ef.dif.psi(ef.dif.h))>pi/4);ef.dif.TEh=ef.dif.h(ef.dif.TEh);ef.dif.TMh=ef.dif.h(ef.dif.TMh);
[ef.dif.TEb,ef.dif.TMb]=retfind(abs(ef.dif.psi(ef.dif.b))>pi/4);ef.dif.TEb=ef.dif.b(ef.dif.TEb);ef.dif.TMb=ef.dif.b(ef.dif.TMb);


% normalisation des 'phases'
% incidents
c=zeros(1,size(ef.inc.E,2));
fe=ef.inc.TE;fm=ef.inc.TM;
c(fe)=abs(ef.inc.E(2,fe))./ef.inc.E(2,fe);
c(fm)=abs(ef.inc.H(2,fm))./ef.inc.H(2,fm);
cc=diag(c);
ef.inc.E=ef.inc.E*cc;
ef.inc.H=ef.inc.H*cc;
ef.inc.champ=ef.inc.champ*cc;
ef.amplitude=ef.amplitude*cc;

% diffractes
c=zeros(1,size(ef.dif.E,2));
fe=ef.dif.TE;fm=ef.dif.TM;
c(fe)=abs(ef.dif.E(2,fe))./ef.dif.E(2,fe);
c(fm)=abs(ef.dif.H(2,fm))./ef.dif.H(2,fm);
cc=diag(c);
ef.dif.E=ef.dif.E*cc;
ef.dif.H=ef.dif.H*cc;
ef.dif.champ=ef.dif.champ*cc;
ef.amplitude=diag(1./c)*ef.amplitude;

% construction de zordre
ef.inc.zordre=ef.inc.ordre(1,:)+i*ef.inc.ordre(2,:);
ef.dif.zordre=ef.dif.ordre(1,:)+i*ef.dif.ordre(2,:);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else;                             %%%%%%%%%%%%%
                                  %    1 D    %
                                  %%%%%%%%%%%%%

cm=[cb(:,1),-i*cb(:,2),zeros(size(cb(:,1)))];% E Hx Hy
cp=[cb(:,1),i*cb(:,2),zeros(size(cb(:,1)))];% E->E   H-> -H

poldb={anglb{1}(difb),anglb{8}(difb,:),anglb{9}(difb,:),betb(difb,:),cm(difb,:)};% diffractes en bas
%        teta            eem               hhm              beta          champ
polib={anglb{1}(incb),anglb{4}(incb,:),anglb{5}(incb,:),betb(incb,:),cp(incb,:)};% incidents en bas
%        teta            eep               hhp             beta         champ
cm=[ch(:,1),-i*ch(:,2),zeros(size(ch(:,1)))];% E Hx Hy
cp=[ch(:,1),i*ch(:,2),zeros(size(ch(:,1)))];% E->E   H-> -H

polih={anglh{1}(inch),anglh{8}(inch,:),anglh{9}(inch,:),beth(inch,:),cm(inch,:)};% incidents en haut
%        teta            eem               hhm             beta        champ
poldh={anglh{1}(difh),anglh{4}(difh,:),anglh{5}(difh,:),beth(difh,:),cp(difh,:)};% diffractes en haut
%        teta            eep               hhp             beta         champ

s=reteval(s);


% construction de la structure 'conviviale' ef
 inc=struct('teta',[polib{1}.',polih{1}.'],'beta',[polib{4}.',polih{4}.'],...
'ordre',[],'E',[polib{2}.',polih{2}.'],'H',[polib{3}.',polih{3}.'],'champ',[polib{5}.',polih{5}.'],...
'sens',[ones(1,length(polib{1})),-ones(1,length(polih{1}))],'h',[],'b',[]);
inc.ordre=round((inc.beta(1,:)-beta0)*d/(2*pi));

dif=struct('teta',[poldh{1}.',poldb{1}.'],'beta',[poldh{4}.',poldb{4}.'],...
'ordre',[],'E',[poldh{2}.',poldb{2}.'],'H',[poldh{3}.',poldb{3}.'],'champ',[poldh{5}.',poldb{5}.'],...
'sens',[ones(1,length(poldh{1})),-ones(1,length(poldb{1}))],'h',[],'b',[]);
dif.ordre=round((dif.beta(1,:)-beta0)*d/(2*pi));

ef=struct('inc',inc,'dif',dif,'amplitude',s);

if isempty(ef.inc.ordre);return;end;

% complement de ef.dif dans le cas de symetries
if ~isempty(sym);
if sym(1)~=0;  % symetrie par rapport a l'axe y
k=find(ef.dif.ordre~=0);   
ef.amplitude(k,:)=ef.amplitude(k,:)/sqrt(2);    
ef.amplitude=[ef.amplitude;diag(sym(1).*exp(2i*ef.dif.beta(1,k)*sym(2)))*ef.amplitude(k,:)];    
ef.dif.teta=[ef.dif.teta,-ef.dif.teta(k)];    
ef.dif.beta=[ef.dif.beta,-ef.dif.beta(k)];    
ef.dif.ordre=[ef.dif.ordre,-ef.dif.ordre(1,k)];    
ef.dif.champ(:,k)=ef.dif.champ(:,k)*sqrt(2); 
ef.dif.champ=[ef.dif.champ,diag([1,1,-1])*ef.dif.champ(:,k)];    
ef.dif.E(:,k)=ef.dif.E(:,k)*sqrt(2);ef.dif.H(:,k)=ef.dif.H(:,k)*sqrt(2);    
ef.dif.E=[ef.dif.E,ef.dif.E(k)];    
ef.dif.H=[ef.dif.H,ef.dif.H(k)]; 
ef.dif.sens=[ef.dif.sens,ef.dif.sens(k)];    
end;  
    
end;



% composantes en y de champ
ef.inc.champ(3,:)=-ef.inc.H.*sin(ef.inc.teta);
ef.dif.champ(3,:)=-ef.dif.H.*sin(ef.dif.teta);

% remplissage de 'h' et 'b'
[ef.inc.b,ef.inc.h]=retfind(ef.inc.sens>0);
[ef.dif.h,ef.dif.b]=retfind(ef.dif.sens>0);   



% mise en ordre par prioritee: sens ,ordre
[prv,nnh]=sort(ef.dif.ordre(ef.dif.h));
[prv,nnb]=sort(ef.dif.ordre(ef.dif.b));
nn=[ef.dif.h(nnh),ef.dif.b(nnb)];ef.dif.h=1:length(nnh);ef.dif.b=length(nnh)+1:length(nnh)+length(nnb);

ef.dif.teta=ef.dif.teta(nn);
ef.dif.beta=ef.dif.beta(:,nn);
ef.dif.ordre=ef.dif.ordre(:,nn);
ef.dif.E=ef.dif.E(:,nn);
ef.dif.H=ef.dif.H(:,nn);
ef.dif.champ=ef.dif.champ(:,nn);
ef.dif.sens=ef.dif.sens(nn);
ef.amplitude=ef.amplitude(nn,:);

[prv,nnh]=sort(ef.inc.ordre(ef.inc.h));
[prv,nnb]=sort(ef.inc.ordre(ef.inc.b));
nn=[ef.inc.b(nnb),ef.inc.h(nnh)];ef.inc.b=1:length(nnb);ef.inc.h=length(nnb)+1:length(nnb)+length(nnh);

ef.inc.teta=ef.inc.teta(nn);
ef.inc.beta=ef.inc.beta(:,nn);
ef.inc.ordre=ef.inc.ordre(:,nn);
ef.inc.E=ef.inc.E(:,nn);
ef.inc.H=ef.inc.H(:,nn);
ef.inc.champ=ef.inc.champ(:,nn);
ef.inc.sens=ef.inc.sens(nn);
ef.amplitude=ef.amplitude(:,nn);


% normalisation des 'phases'
% incidents
c=abs(ef.inc.E)./ef.inc.E;
cc=diag(c);
ef.inc.E=ef.inc.E*cc;
ef.inc.H=ef.inc.H*cc;
ef.inc.champ=ef.inc.champ*cc;
ef.amplitude=ef.amplitude*cc;

% diffractes
c=abs(ef.dif.E)./ef.dif.E;
cc=diag(c);
ef.dif.E=ef.dif.E*cc;
ef.dif.H=ef.dif.H*cc;
ef.dif.champ=ef.dif.champ*cc;
ef.amplitude=diag(1./c)*ef.amplitude;
                                
end;            % 2 D 1 D                                  
                                  
