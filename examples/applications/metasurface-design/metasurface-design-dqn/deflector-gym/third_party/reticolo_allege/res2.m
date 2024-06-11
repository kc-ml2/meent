function [ef,TAB,sh,sb]=res2(aa,PROFIL,parm);
%  function [ef,TAB,sh,sb]=res2(aa,PROFIL,parm);
%  aa calcule precedemment
%
% PROFIL ={ {haut1,seq1,fois1} ,{haut2,seq2,fois2} ,...{hautn,seqn,foisn} }; tableau de cell decrivant des modules repetes
% ou            PROFIL ={hauteur,sequence,fois}
% ou            PROFIL ={hauteur,sequence}
%
%
%   hauteurs: hauteurs (unites metriques) de haut en bas   
%   sequence:sequence de textures correspondantes  
%   fois repetitions du module( facultatif  par defaut fois=1)
%   si une hauteur est nulle il n'y a pas calcul sur la couche associee
%   ( on peut donc simplifier un profil en mettant certaines hauteurs nulles sans augmenter le temps de calcul)  
%
%  la premiere texture est le superstrat (avec eventuellement une hauteur 0)
%  la derniere texture est le substrat (avec eventuellement une hauteur 0)
%
%  le tableau TAB donne les hauteurs et les numeros de texture associes (utile pour verification)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  res2 construit la structure ef  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%          %%%%%%%%%%%%%%%%%
%          %     2 D       %
%          %  ou conique   %
%          %%%%%%%%%%%%%%%%%
%
%     certains tableaux peuvent etre vides  (notamment a cause des symetries)
%    si parm.res2.=1 (option par defaut) ef est de la forme:
%
% 
%                           TEinc_top: [1x1 struct]
%                 TEinc_top_reflected: [1x1 struct]
%               TEinc_top_transmitted: [1x1 struct]
% 
%                        TEinc_bottom: [1x1 struct]
%              TEinc_bottom_reflected: [1x1 struct]
% ef         TEinc_bottom_transmitted: [1x1 struct]
% 
%                           TMinc_top: [1x1 struct]
%                 TMinc_top_reflected: [1x1 struct]
%               TMinc_top_transmitted: [1x1 struct]
% 
%                        TMinc_bottom: [1x1 struct]
%              TMinc_bottom_reflected: [1x1 struct]
%            TMinc_bottom_transmitted: [1x1 struct]
%  
% 
% 
%                                  
% avec par exemple:
%                          theta: [5x1 double]
%                          delta: [5x1 double]
%                              K: [5x3 double] vecteur d'onde unitaire
%
%                     efficiency: [5x1 double]
%                  efficiency_TE: [5x1 double]
%                  efficiency_TM: [5x1 double]
%                   amplitude_TE: [5x1 double]
%                   amplitude_TM: [5x1 double]
%
%  ef.TEinc_top_transmitted    E: [5x3 double]  composantes de E a l'origine 
%                              H: [5x3 double]  composantes de H a l'origine 
%
%                 PlaneWave_TE_E: [5x3 double] composantes de l'onde plane TE  (TM)
%                 PlaneWave_TE_H: [5x3 double] dont la composante du vecteur de 
%                                              poynting selon z est 1/2
%
%                PlaneWave_TE_Eu: [5x2 double]  memes ondes 
%                PlaneWave_TE_Hu: [5x2 double]  dans le repere u_TM ,u_TE
%
%                 PlaneWave_TM_E: [5x3 double]
%                 PlaneWave_TM_H: [5x3 double]
%
%                PlaneWave_TM_Eu: [5x2 double]
%
%          %%%%%%%%%%%%%%%%%
%          %     1 D       %
%          %%%%%%%%%%%%%%%%%
%
%
% 
%                           inc_top: [1x1 struct]
%                inc_top_reflected: [1x1 struct]
%               inc_top_transmitted: [1x1 struct]
% ef
%                       inc_bottom: [1x1 struct]
%              inc_bottom_reflected: [1x1 struct]
%           inc_bottom_transmitted: [1x1 struct]
% 
%  
% 
% 
%                                  
% avec par exemple:
%                          theta: [5x1 double]
%                              K: [5x3 double] vecteur d'onde unitaire
%
%                     efficiency: [5x1 double]
%                     amplitude: [5x1 double]
%
%  ef.inc_top_transmitted      E: [5x3 double]  composantes de E a l'origine 
%                              H: [5x3 double]  composantes de H a l'origine 
%
%                    PlaneWave_E: [5x3 double] composantes de l'onde plane TE  (TM)
%                    PlaneWave_H: [5x3 double] dont la composante du vecteur de 
%                                              poynting selon z est 1/2




%   si parm.res2.result=0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                 inc:  [1x1 struct]            ordres incidents                                 %%%%%%%%%%%%%%%%%
%       ef.       dif: [1x1 struct]           ordres diffractes                                  %     2 D       %
%                amplitude: [46x4 double]    amplitudes diffractees                              %  ou conique   %
%                result:                     forme 'simplifiee' en anglais                       %%%%%%%%%%%%%%%%%
%
% 
%           teta: [6.6478 6.6478 10.0000 10.0000]       angles teta(ligne)
%            delta: [20 20 20.0000 20.0000]   angles delta(ligne)
%              psi: [90 0 90 0]  angles psi(ligne)
%             beta: [2x4 double]   beta composantes x y du vecteur d'onde
%            ordre: [2x4 double]   ordres de diffraction
%           zordre: [1x4 double]   ordres de diffraction en complexe(commode a utiliser pour chercher un ordre donne)
%                E: [2x4 double]  composantes de E dans le repere u v (ligne 1:composante sur u   ligne 2:composante sur v)
%                H: [2x4 double]  composantes de H dans le repere u v
%            champ: [6x4 double]  [Ex;Ey;Ez;Hx;Hy;Hz] a l'origine
%             sens: [1 1 -1 -1]   sens de propagation 1:vers le haut  -1 vers le bas
%                h: [3 4]  numeros des incidents du haut
%   ef.inc.      b: [1 2]  numeros des incidents du bas
%               TE: [1 3]   numeros des incidents TE
%               TM: [2 4]   numeros des incidents TM
%              TEh: 3  numeros des incidents  du haut TE    \
%              TMh: 4  numeros des incidents du haut TM      |        en general un element unique 
%              TEb: 1  numeros des incidents  du bas TE      |  mais certains peuvent etre absents a cause des symetries
%              TMb: 2  numeros des incidents  du bas TM     /
%
%
%
%
%            teta: [1x46 double]      angles teta(ligne)
%            delta: [1x46 double]    angles delta(ligne)
%              psi: [1x46 double]   angles psi(ligne)
%             beta: [2x46 double]   beta composantes x y du vecteur d'onde
%            ordre: [2x46 double]   ordres de diffraction
%           zordre: [1x46 double]   ordres de diffraction en complexe(commode a utiliser pour chercher un ordre donne)
%                E: [2x46 double]   composantes de E dans le repere u v
%                H: [2x46 double]   composantes de H dans le repere u v
%            champ: [6x46 double] [Ex;Ey;Ez;Hx;Hy;Hz] a l'origine
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
%     l' amplitude de ces ondes planes est normalisee proportionnellement au flux d'energie a travers le plan x y
%       (les champs E et H d'une onde normale a ce plan dans un milieu d'indice 1 etant E=1  H=1 )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   les autres elements de ef donnent un description 'plus directe' 
%
%                                 ordre:
%                                zordre: 
%                                  teta:
%                                 delta:   *******************************
%   ef.TEh  .  inc   .    efficacite_TE:   *  onde TE incidente du haut  *
%                         efficacite_TM:   *******************************    
%                            efficacite: 
%                                 champ: 
%                                     E: 
%                                     H:
%
%                                 ordre:
%                                zordre:    *********************************************************
%                                  teta:    *   diffractes en haut par l'onde TE incidente du haut  *
%                                 delta:    *********************************************************
%   ef.TEh  .  haut  .    efficacite_TE:   efficacite de la composante TE
%                         efficacite_TM:   efficacite de la composante TM
%                            efficacite:   efficacite totale
%                                 champ:   6 composantes du champ(en colonne) a l'origine
%                                     E:   composantes de E a l'origine dans le repere u v 
%                                     H:   composantes de H a l'origine dans le repere u v 
%
%                                 ordre:
%                                zordre: 
%                                  teta:
%                                 delta: 
%   ef.TEh  .  bas   .    efficacite_TE:   *********************************************************
%                         efficacite_TM:   *   diffractes en haut par l'onde TE incidente du haut  *
%                            efficacite:   *********************************************************
%                                 champ:  6 composantes du champ(en colonne) a l'origine
%                                     E:  composantes de E a l'origine dans le repere u v 
%                                     H:  composantes de H a l'origine dans le repere u v 
%
%
%  et  pareil  pour     ef.TEb  onde TE incidente du bas
%                       ef.TMh  onde TM incidente du haut
%                       ef.TMb  onde TM incidente du bas
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
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
%
%      par exemple:
%      les efficacitees diffractees par l'onde TE incidente du haut sur les ondes TM en bas
%      forment le vecteur colonne:   abs(  ef.amplitude(ef.dif.TMb,ef.inc.TEh) ).^2  
%
%      la somme des efficacitees diffractees par l'onde TM incidente du bas dans tout l'espace est
%             sum(  abs(ef.amplitude(:,ef.inc.TMb)).^2 )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                 inc:           ordres incidents                                 %%%%%%%%%%%%%%%%%
%       ef.       dif:         ordres diffractes                                  %     1 D       %
%                amplitude:   amplitudes diffractees                              %%%%%%%%%%%%%%%%ù
% 
%             teta:        angles teta(ligne)
%             beta: [2x4 double]   beta composantes x  du vecteur d'onde
% ef.inc.    ordre:   ordre de diffraction
%            champ:  a l'origine   [Ez;Hx;Hy] en TE    [Hz;Ex;Ey] en TM
%             sens:   sens de propagation 1:vers le haut  -1 vers le bas
%                h:  numero de l'incident du haut
%                b:  numero de l'incident du bas
%
%             teta:       angles teta(ligne)
%             beta:   beta composantes x du vecteur d'onde
% ef.dif.    ordre:    ordres de diffraction
%            champ:  a l'origine   [Ez;Hx;Hy] en TE    [Hz;Ex;Ey] en TM
%             sens:    sens de propagation 1:vers le haut  -1 vers le bas
%                h:   numeros des diffractes en haut
%                b:   numeros des diffractes en bas
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
%


aa=retio(aa);
if nargin<3;parm=res0;end;
if ~iscell(PROFIL{1});PROFIL={PROFIL};end;
ih=PROFIL{1}{2}(1);ib=PROFIL{end}{2}(end); % textures en haut et en bas;

a=aa{1};init=aa{2};n=aa{3};UN=aa{4};D=aa{5};beta0=aa{6};sym=aa{7};
if init{end}.sog;retss(parm.res2.retss);else;retss(parm.res2.retgg);end;

TAB=zeros(0,2);
if parm.res2.cale==1; % on ne calcule sh et sb que si on veut calculer e
[sb,bas,betb,cb,anglb]=retb(init,a{ib},-parm.res2.tolb);
if isempty(betb);incb=[];else;
incb=find(abs(sum(abs(betb-repmat(beta0,size(betb,1),1)),2))<eps);end;%  incidents (eventuellement 2 polarisations)

sb=rettronc(sb,bas(incb,1),bas(:,1),-1);
end;

if parm.res2.cals==1;s=sb;end;
for module=size(PROFIL,2):-1:1;  % on integre de bas en haut
sequence=PROFIL{module}{2};ntextures=length(sequence);    
HAUTEURS=PROFIL{module}{1};
if size(PROFIL{module},2)>=3;fois=PROFIL{module}{3};else;fois=1;end;

if fois==1;  % le module n'est pas repete
for ii=ntextures:-1:1;
TAB=[[HAUTEURS(ii),sequence(ii)];TAB];
if HAUTEURS(ii)~=0 & parm.res2.cals==1;
s=retss(retc(a{sequence(ii)},HAUTEURS(ii)/UN),s);end;
end;
else;         % le module est repete fois
premier=1;TABB=[];    
for ii=ntextures:-1:1;
TABB=[[HAUTEURS(ii),sequence(ii)];TABB];    
if HAUTEURS(ii)~=0 & parm.res2.cals==1;
if premier==1;premier=0;ss=retc(a{sequence(ii)},HAUTEURS(ii)/UN);else;ss=retss(retc(a{sequence(ii)},HAUTEURS(ii)/UN),ss);end;  
end;
end;
if premier==0 & parm.res2.cals==1;s=retss(retsp(ss,fois),s);end;
TAB=[repmat(TABB,fois,1);TAB];
end;
end;          % fin module

if parm.res2.cale~=1;sh=[];sb=[];ef=[];return;end;% on ne calcule sh et sb que si on veut calculer e 
[sh,haut,beth,ch,anglh]=retb(init,a{ih},parm.res2.tolh);
if isempty(beth);inch=[];else;
inch=find(abs(sum(abs(beth-repmat(beta0,size(beth,1),1)),2))<eps);end;%  incidents(eventuellement 2 polarisations)

sh=rettronc(sh,haut(inch,1),haut(:,1),1);


if parm.res2.cals==1;s=retss(sh,s);else;s=[];end;
if parm.res2.calef==0;
if init{end}.dim==2;ef=struct('Einch',anglh{15}(inch,:).','Eincb',anglb{7}(incb,:).','Hinch',anglh{16}(inch,:).','Hincb',anglb{8}(incb,:).');  % forme simplifiee de ef
%                                           eem                        eep                          hhm                          hhp
else;
ef=struct('Einch',anglh{8}(inch).','Eincb',anglb{4}(incb).','Hinch',anglh{9}(inch).','Hincb',anglb{5}(incb).');end;  % 1 D 
return;end;

ef=retreseau(init,s,betb,cb,anglb,incb,0,beth,ch,anglh,inch,0);

% conversion des angles en degres
ef.inc.teta=(180/pi)*ef.inc.teta;ef.dif.teta=(180/pi)*ef.dif.teta;
if init{end}.dim==2;
ef.inc.delta=(180/pi)*ef.inc.delta;ef.inc.psi=(180/pi)*ef.inc.psi;
ef.dif.delta=(180/pi)*ef.dif.delta;ef.dif.psi=(180/pi)*ef.dif.psi;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% autre forme de ef en 2D

if init{end}.dim==2; %  <------------autre forme de ef en 2D
ef=complete(ef);  
if abs(parm.res2.result)==1 ;ef=tri2D(ef);ef=reticolo(ef);end;% structure plus simple en anglais de classe 'reticolo1D'

else;% <-------------- mise en forme  en 1 D
if abs(parm.res2.result)==1; % <++ structure plus simple en anglais

% construction du ef 2 D
if aa{end}==0;pol=1;else;pol=-1;end; 
ef.inc=deuxd(ef.inc,pol);
ef.dif=deuxd(ef.dif,pol);
ef=complete(ef); 
ef=tri2D(ef);

else; %    <++

ef.inc=rmfield(ef.inc,{'E','H'});ef.dif=rmfield(ef.dif,{'E','H'});   
if aa{end}==0;% E//
ef.pol='TE'; 
else;       % H//
ef.pol='TM'; 
ef.inc.champ(2:3,:)=-ef.inc.champ(2:3,:);
ef.dif.champ(2:3,:)=-ef.dif.champ(2:3,:);
end;
% mise en forme des elements vides en 1 D
ef.inc=zer1D(ef.inc);ef.dif=zer1D(ef.dif);
end; %     <++
end; %  <------------
% % TEST
% 
% if abs(parm.res2.result)==1;
% persistent er;if isempty(er);er=0;end;
% testef=inline(['sum(sum(abs(diag(a.amplitude_TE)*a.PlaneWave_TE_E+diag(a.amplitude_TM)*a.PlaneWave_TM_E-a.E)))'...
% '+sum(sum(abs(diag(a.amplitude_TE)*a.PlaneWave_TE_H+diag(a.amplitude_TM)*a.PlaneWave_TM_H-a.H)))'],'a');
% % verification de E H=sum( PlaneWave * amplitude)
% err=testef(ef.TEinc_top)+testef(ef.TEinc_top_reflected)+testef(ef.TEinc_top_transmitted)...
% +testef(ef.TEinc_bottom)+testef(ef.TEinc_bottom_reflected)+testef(ef.TEinc_bottom_transmitted)...
% +testef(ef.TMinc_top)+testef(ef.TMinc_top_reflected)+testef(ef.TMinc_top_transmitted)...
% +testef(ef.TMinc_bottom)+testef(ef.TMinc_bottom_reflected)+testef(ef.TMinc_bottom_transmitted);
% % verification du flux de poynting sur z
% [pE,pM,errr]=calpz(ef.TEinc_top);err=errr+err+sum(abs(pE+1));
% [pE,pM,errr]=calpz(ef.TMinc_top);err=errr+err+sum(abs(pM+1));
% [pE,pM,errr]=calpz(ef.TEinc_bottom);err=errr+err+sum(abs(pE-1));
% [pE,pM,errr]=calpz(ef.TMinc_bottom);err=errr+err+sum(abs(pM-1));
% 
% [pE,pM,errr]=calpz(ef.TEinc_top_reflected);err=errr+err+sum(abs(pE-1))+sum(abs(pM-1));
% [pE,pM,errr]=calpz(ef.TMinc_top_reflected);err=errr+err+sum(abs(pE-1))+sum(abs(pM-1));
% [pE,pM,errr]=calpz(ef.TEinc_top_transmitted);err=errr+err+sum(abs(pE+1))+sum(abs(pM+1));
% [pE,pM,errr]=calpz(ef.TMinc_top_transmitted);err=errr+err+sum(abs(pE+1))+sum(abs(pM+1));
% 
% [pE,pM,errr]=calpz(ef.TEinc_bottom_reflected);err=errr+err+sum(abs(pE+1))+sum(abs(pM+1));
% [pE,pM,errr]=calpz(ef.TMinc_bottom_reflected);err=errr+err+sum(abs(pE+1))+sum(abs(pM+1));
% [pE,pM,errr]=calpz(ef.TEinc_bottom_transmitted);err=errr+err+sum(abs(pE-1))+sum(abs(pM-1));
% [pE,pM,errr]=calpz(ef.TMinc_bottom_transmitted);err=errr+err+sum(abs(pE-1))+sum(abs(pM-1));
% 
% er=er+err
% if er>.1;
% 1;
% end;
% end;
% 
% 
% % FIN TEST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (init{end}.dim==1)&(parm.res2.result==1);% <***' degraissage' de ef
%fields=fieldnames(ef);for ii=1:length(fields);if isempty(getfield(ef,fields{ii},'order'));ef=rmfield(ef,fields{ii});end;end;
fields=fieldnames(ef);
if aa{end}==0;ef=rmfield(ef,fields([7:12]));fields=fields(1:6);else;ef=rmfield(ef,fields([1:6]));fields=fields(7:12);end;% E// ou H//

for ii=1:length(fields);
eff=getfield(ef,fields{ii});
if aa{end}==0;% E//
eff=struct('order',eff.order(:,1),'theta',eff.theta,'K',eff.K,'efficiency',eff.efficiency_TE,...
'amplitude',eff.amplitude_TE,'E',eff.E,'H',eff.H,'PlaneWave_E',eff.PlaneWave_TE_E,'PlaneWave_H',eff.PlaneWave_TE_H);
else;   % H//
eff=struct('order',eff.order(:,1),'theta',eff.theta,'K',eff.K,'efficiency',eff.efficiency_TM,...
'amplitude',eff.amplitude_TM,'E',eff.E,'H',eff.H,'PlaneWave_E',eff.PlaneWave_TM_E,'PlaneWave_H',eff.PlaneWave_TM_H);
end;
ef=rmfield(ef,fields{ii});ef=setfield(ef,fields{ii}(3:end),eff);   
end;
ef=reticolo(ef); % transformation en classe 'reticolo1D'
end;  % <*** degraissage
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 
% % TEST
% 
% function [pzE,pzM,er]=calpz(a);er=0;pzE=[];pzM=[];if isempty(a.order);return;end;
% E=a.PlaneWave_TE_E;H=a.PlaneWave_TE_H;pzE=E(:,1).*conj(H(:,2))-E(:,2).*conj(H(:,1));
% pxE=E(:,2).*conj(H(:,3))-E(:,3).*conj(H(:,2));
% pyE=E(:,3).*conj(H(:,1))-E(:,1).*conj(H(:,3));
% E=a.PlaneWave_TM_E;H=a.PlaneWave_TM_H;pzM=E(:,1).*conj(H(:,2))-E(:,2).*conj(H(:,1));
% pxM=E(:,2).*conj(H(:,3))-E(:,3).*conj(H(:,2));
% pyM=E(:,3).*conj(H(:,1))-E(:,1).*conj(H(:,3));
% 
% KE=a.K(pzE~=0,:);KM=a.K(pzM~=0,:);
% pxE=pxE(pzE~=0);pxM=pxM(pzM~=0);
% pyE=pyE(pzE~=0);pyM=pyM(pzM~=0);
% pzE=pzE(pzE~=0);pzM=pzM(pzM~=0);
% pE=[pxE,pyE,pzE];pE=diag(1./sqrt(sum(pE.^2,2)))*pE;
% pM=[pxM,pyM,pzM];pM=diag(1./sqrt(sum(pM.^2,2)))*pM;
% v=[-sin(a.delta*pi/180),cos(a.delta*pi/180),0*a.delta];
% u=[a.K(:,3).*v(:,2),-a.K(:,3).*v(:,1),a.K(:,2).*v(:,1)-a.K(:,1).*v(:,2)];
% er=0;
% er=max(er,retcompare(KE,pE));
% er=max(er,retcompare(KM,pM));
% er=max(er,retcompare(diag(a.PlaneWave_TE_Eu(:,1))*u+diag(a.PlaneWave_TE_Eu(:,2))*v,a.PlaneWave_TE_E));
% er=max(er,retcompare(diag(a.PlaneWave_TE_Hu(:,1))*u+diag(a.PlaneWave_TE_Hu(:,2))*v,a.PlaneWave_TE_H));
% er=max(er,retcompare(diag(a.PlaneWave_TM_Eu(:,1))*u+diag(a.PlaneWave_TM_Eu(:,2))*v,a.PlaneWave_TM_E));
% er=max(er,retcompare(diag(a.PlaneWave_TM_Hu(:,1))*u+diag(a.PlaneWave_TM_Hu(:,2))*v,a.PlaneWave_TM_H));
% if er>.1;
% 1
% end;
% % FIN TEST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function a=tri(ef,kinc,kdif);
zzordre=retelimine(ef.dif.zordre(kdif));
ordre=[];zordre=[];teta=[];delta=[];amplitude_TE=[];amplitude_TM=[];efficacite=[];champ=[];E=[];H=[];
champ_TE=[];champ_TM=[];champ_TE_uv=[];champ_TM_uv=[];
if ~isempty(kinc);
for ii=1:length(zzordre);
k=kdif(find(ef.dif.zordre(kdif)==zzordre(ii)));
eff=ef.amplitude(k,kinc);
effTE=0;effTM=0;ampTE=0;ampTM=0;champTE=zeros(6,1);champTM=zeros(6,1);champTEuv=zeros(4,1);champTMuv=zeros(4,1);
for jj=1:length(k);
if ismember(k(jj),ef.dif.TE);ampTE=eff(jj);champTE=ef.dif.champ(:,k(jj));champTEuv=[ef.dif.E(:,k(jj));ef.dif.H(:,k(jj))];end;
if ismember(k(jj),ef.dif.TM);ampTM=eff(jj);champTM=ef.dif.champ(:,k(jj));champTMuv=[ef.dif.E(:,k(jj));ef.dif.H(:,k(jj))];end;
end;
champ_TE=[champ_TE,champTE];
champ_TM=[champ_TM,champTM];
champ_TE_uv=[champ_TE_uv,champTEuv];
champ_TM_uv=[champ_TM_uv,champTMuv];
amplitude_TE=[amplitude_TE,ampTE];
amplitude_TM=[amplitude_TM,ampTM];
efficacite=[efficacite,sum(abs(eff).^2)];
champ=[champ,ef.dif.champ(:,k)*eff];
E=[E,ef.dif.E(:,k)*eff];
H=[H,ef.dif.H(:,k)*eff];
ordre=[ordre,ef.dif.ordre(:,k(1))];
zordre=[zordre,ef.dif.zordre(k(1))];
teta=[teta,ef.dif.teta(k(1))];
delta=[delta,ef.dif.delta(k(1))];
end;
end;
a=struct('ordre',ordre,'zordre',zordre,'teta',teta,'delta',delta,...
'amplitude_TE',amplitude_TE,'amplitude_TM',amplitude_TM,...
'efficacite_TE',abs(amplitude_TE).^2,'efficacite_TM',abs(amplitude_TM).^2,'efficacite',efficacite,...
'champ_TE',champ_TE,'champ_TM',champ_TM,'champ_TE_uv',champ_TE_uv,'champ_TM_uv',champ_TM_uv,'champ',champ,'E',E,'H',H);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=zer(a); % mise en forme des elements vides
if isempty(a.teta);a.teta=zeros(1,0);end;
if isempty(a.delta);a.delta=zeros(1,0);end;
if isempty(a.psi);a.psi=zeros(1,0);end;
if isempty(a.ordre);a.ordre=zeros(2,0);end;
if isempty(a.zordre);a.zordre=zeros(1,0);end;
if isempty(a.E);a.E=zeros(2,0);end;
if isempty(a.H);a.H=zeros(2,0);end;
if isempty(a.champ);a.champ=zeros(6,0);end;
if isempty(a.sens);a.sens=zeros(1,0);end;
if isempty(a.h);a.h=zeros(1,0);end;
if isempty(a.b);a.b=zeros(1,0);end;
if isempty(a.TE);a.TE=zeros(1,0);end;
if isempty(a.TM);a.TM=zeros(1,0);end;
if isempty(a.TEh);a.TEh=zeros(1,0);end;
if isempty(a.TMh);a.TMh=zeros(1,0);end;
if isempty(a.TEb);a.TEb=zeros(1,0);end;
if isempty(a.TMb);a.TMb=zeros(1,0);end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=zer1(a); % mise en forme des elements vides
if isempty(a.ordre);a.ordre=zeros(2,0);end;
if isempty(a.zordre);a.zordre=zeros(1,0);end;
if isempty(a.teta);a.teta=zeros(1,0);end;
if isempty(a.delta);a.delta=zeros(1,0);end;
if isempty(a.amplitude_TE);a.amplitude_TE=zeros(1,0);end;
if isempty(a.amplitude_TM);a.amplitude_TM=zeros(1,0);end;
if isempty(a.efficacite_TE);a.efficacite_TE=zeros(1,0);end;
if isempty(a.efficacite_TM);a.efficacite_TM=zeros(1,0);end;
if isempty(a.efficacite);a.efficacite=zeros(1,0);end;
if isempty(a.champ_TE);a.champ_TE=zeros(6,0);end;
if isempty(a.champ_TM);a.champ_TM=zeros(6,0);end;
if isempty(a.champ_TE_uv);a.champ_TE_uv=zeros(4,0);end;
if isempty(a.champ_TM_uv);a.champ_TM_uv=zeros(4,0);end;
if isempty(a.champ);a.champ=zeros(6,0);end;
if isempty(a.E);a.E=zeros(2,0);end;
if isempty(a.H);a.H=zeros(2,0);end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=zer1D(a); % mise en forme des elements vides en 1 D
if isempty(a.teta);a.teta=zeros(1,0);end;
if isempty(a.ordre);a.ordre=zeros(1,0);end;
if isempty(a.champ);a.champ=zeros(3,0);end;
if isempty(a.sens);a.sens=zeros(1,0);end;
if isempty(a.h);a.h=zeros(1,0);end;
if isempty(a.b);a.b=zeros(1,0);end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result=tri2D(ef);
% Tri de la structure ef en 2 D, réarrangement en une structure plus simple et en anglais

TEinc_top=trans(ef.TEh.inc,-1);TEinc_top_reflected=trans(ef.TEh.haut,1);TEinc_top_transmitted=trans(ef.TEh.bas,-1);
TEinc_bottom=trans(ef.TEb.inc,1);TEinc_bottom_reflected=trans(ef.TEb.bas,-1);TEinc_bottom_transmitted=trans(ef.TEb.haut,1);

TMinc_top=trans(ef.TMh.inc,-1);TMinc_top_reflected=trans(ef.TMh.haut,1);TMinc_top_transmitted=trans(ef.TMh.bas,-1);
TMinc_bottom=trans(ef.TMb.inc,1);TMinc_bottom_reflected=trans(ef.TMb.bas,-1);TMinc_bottom_transmitted=trans(ef.TMb.haut,1);

result=struct('TEinc_top',TEinc_top,'TEinc_top_reflected',TEinc_top_reflected,'TEinc_top_transmitted',TEinc_top_transmitted,...
'TEinc_bottom',TEinc_bottom,'TEinc_bottom_reflected',TEinc_bottom_reflected,'TEinc_bottom_transmitted',TEinc_bottom_transmitted,...
'TMinc_top',TMinc_top,'TMinc_top_reflected',TMinc_top_reflected,'TMinc_top_transmitted',TMinc_top_transmitted,...
'TMinc_bottom',TMinc_bottom,'TMinc_bottom_reflected',TMinc_bottom_reflected,'TMinc_bottom_transmitted',TMinc_bottom_transmitted);

%............................
function a=trans(a,sens);
a=struct('order',[real(a.zordre).',imag(a.zordre).'],'theta',real(a.teta).','delta',real(a.delta).',...
'K',[(sin(a.teta*pi/180).*cos(a.delta*pi/180)).',(sin(a.teta*pi/180).*sin(a.delta*pi/180)).',sens*cos(a.teta*pi/180).'],...
'efficiency',a.efficacite.','efficiency_TE',a.efficacite_TE.','efficiency_TM',a.efficacite_TM.',...
'amplitude_TE',a.amplitude_TE.','amplitude_TM',a.amplitude_TM.','E',a.champ(1:3,:).','H',a.champ(4:6,:).',...
'PlaneWave_TE_E',a.champ_TE(1:3,:).','PlaneWave_TE_H',a.champ_TE(4:6,:).',...
'PlaneWave_TE_Eu',a.champ_TE_uv([1,2],:).','PlaneWave_TE_Hu',a.champ_TE_uv([3,4],:).',...
'PlaneWave_TM_E',a.champ_TM(1:3,:).','PlaneWave_TM_H',a.champ_TM(4:6,:).',...
'PlaneWave_TM_Eu',a.champ_TM_uv([1,2],:).','PlaneWave_TM_Hu',a.champ_TM_uv([3,4],:).');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a2=deuxd(a1,pol);  %   construction du ef 2 D a partir du ef 1 D
n=length(a1.teta);
if pol==1;  %  TE
a2=struct('teta',a1.teta,'delta',zeros(1,n),'psi',90*ones(1,n),'beta',a1.beta,'ordre',a1.ordre,'zordre',a1.ordre,...
'E',[0*a1.E;-a1.E],'H',[a1.H;0*a1.H],'champ',[zeros(1,n);-a1.champ(1,:);zeros(1,n);a1.champ(2,:);zeros(1,n);a1.champ(3,:)],'sens',a1.sens,'h',a1.h,'b',a1.b,...
'TE',1:n,'TM',zeros(1,0),'TEh',a1.h,'TMh',zeros(1,0),'TEb',a1.b,'TMb',zeros(1,0));
else;  %  TM
a2=struct('teta',a1.teta,'delta',zeros(1,n),'psi',zeros(1,n),'beta',a1.beta,'ordre',a1.ordre,'zordre',a1.ordre,...
'E',[-a1.H;0*a1.H],'H',[0*a1.E;-a1.E],'champ',[-a1.champ(2,:);zeros(1,n);-a1.champ(3,:);zeros(1,n);-a1.champ(1,:);zeros(1,n)],'sens',a1.sens,'h',a1.h,'b',a1.b,...
'TE',zeros(1,0),'TM',1:n,'TEh',zeros(1,0),'TMh',a1.h,'TEb',zeros(1,0),'TMb',a1.b);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ef=complete(ef);   
ef.TEh=complete1(ef,ef.inc.TEh,1,0);  
ef.TEb=complete1(ef,ef.inc.TEb,1,0);
ef.TMh=complete1(ef,ef.inc.TMh,0,1); 
ef.TMb=complete1(ef,ef.inc.TMb,0,1); 

% mise en forme des elements vides
ef.inc=zer(ef.inc);ef.dif=zer(ef.dif);
ef.TEh.inc=zer1(ef.TEh.inc);ef.TEh.haut=zer1(ef.TEh.haut);ef.TEh.bas=zer1(ef.TEh.bas);
ef.TEb.inc=zer1(ef.TEb.inc);ef.TEb.haut=zer1(ef.TEb.haut);ef.TEb.bas=zer1(ef.TEb.bas);
ef.TMh.inc=zer1(ef.TMh.inc);ef.TMh.haut=zer1(ef.TMh.haut);ef.TMh.bas=zer1(ef.TMh.bas);
ef.TMb.inc=zer1(ef.TMb.inc);ef.TMb.haut=zer1(ef.TMb.haut);ef.TMb.bas=zer1(ef.TMb.bas);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ec=complete1(ef,kinc,te,tm)    

haut=tri(ef,kinc,ef.dif.h);bas=tri(ef,kinc,ef.dif.b);
if isempty(kinc);
inc=struct('ordre',[],'zordre',[],'teta',[],'delta',[],...
'amplitude_TE',[],'amplitude_TM',[],'efficacite_TE',[],'efficacite_TM',[],'efficacite',[],...
'champ_TE',zeros(6,0),'champ_TM',zeros(6,0),'champ_TE_uv',zeros(4,0),'champ_TM_uv',zeros(4,0),'champ',zeros(6,0),'E',zeros(2,0),'H',zeros(2,0));
else;inc=struct('ordre',[0,0],'zordre',0,'teta',ef.inc.teta(kinc),'delta',ef.inc.delta(kinc),...
'amplitude_TE',te,'amplitude_TM',tm,'efficacite_TE',te,'efficacite_TM',tm,'efficacite',1,...
'champ_TE',te*ef.inc.champ(:,kinc),'champ_TM',tm*ef.inc.champ(:,kinc),...
'champ_TE_uv',te*[ef.inc.E(:,kinc);ef.inc.H(:,kinc)],'champ_TM_uv',tm*[ef.inc.E(:,kinc);ef.inc.H(:,kinc)],...
'champ',ef.inc.champ(:,kinc),'E',ef.inc.E(:,kinc),'H',ef.inc.H(:,kinc));
end;
ec=struct('inc',inc,'haut',haut,'bas',bas);









