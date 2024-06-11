function parm=res0(dim);
%function parm=res0;
% parametres par defaut
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  symetries
%  parm.sym.x=x0   x=x0 :plan de symetrie seulement si beta0(1)=0     par defaut parm.sym.x=[]  pas de symetrie en x 
%  parm.sym.y=y0   y=y0 :plan de symetrie seulement si beta0(2)=0     par defaut parm.sym.y=[]  pas de symetrie en y
%  parm.sym.pol=1;   1 TE  -1:TM  par defaut 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  parm.res1.trace:  1  trace des textures     (defaut 0)
%
%  parm.res1.nx    parm.res1.xlimite (valeurs de x et nombre de points pour le trace)
%  parm.res1.ny    parm.res1.ylimite (valeurs de y et nombre de points pour le trace)
%     (par defaut la maille du reseau centree avec 100*100 points)
% exemple:  parm.res1.xlimite=[-D(1),D(1)]; parm.res1.nx=500; 
%
%  parm.res1.angles: pour les ellipses   1 :regulier en angles  0 meilleure repartition reguliers en surface (defaut 1)
%  parm.res1.calcul: 1   calcul  (defaut 1)  (si on ne veut que visualiser les textures prendre parm.res1.calcul=0 )
%  parm.res1.champ:  options pour le trace futur des champs  1: les champs sont bien calcules( gourmand en temps et memoire)
%        0 :les champs non continus sont calcules de facon approchee  et Ez et Hz ne sont pas calcules et remplaces par 0
%                              (defaut 0)
%  parm.res1.ftemp: 1   fichiers temporaires  (defaut 1) 
%  parm.res1.fperm: 'aa'   le resultat est mis sur un fichier permanent  de nom 'aa123..'  (defaut [] donc pas ecriture0) 
%  parm.res1.sog:   1 matrices S    0 matrices G (defaut 1) 
%  parm.res1.li:   sens pour la methode de li (defaut 1 ,la plus rapide dans le cas 1D) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  parm.res2.tolh,  parm.res2.tolb, tolerance pour selectionner les modes propagatifs en haut et en bas (par defaut 1.e-6)
%  parm.res2.cals=1 calcul du produit des matrices S
%  parm.res2.cale=1 si on veut calculer les champs dans res3 ( utilise seulement par res3)
%  parm.res2.calef=1   si on veut calculer ef                ( utilise seulement par res3)
%  parm.res2.retss=3  methode de produit des matrices S
%  parm.res2.retgg=0  methode de produit des matrices G (metaux)
%  parm.res2.result=1  retour par res2 de la structure simplifiee result en anglais et simplification en 1D
%                  -1  retour par res2 de la structure simplifiee result en anglais et non simplification en 1D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  parm.res3.cale=[]     signifie que l'on ne calcule pas le champ,  e contient alors TAB  (utile pour verifier le profil)
%  parm.res3.caltab=1    on ne calcule que tab et meme pas l'objet o
%
%  parm.res3.sens    1:incident du haut   -1  du bas (par defaut 1) 
%
%  parm.res3.npts    nombre de points dans les couches pour le calcul
%        si c'est un tableau de meme longueur que le nombre de couches dans le profil(compte tenu des repetitions eventuelles)
%        chaque element du tableau est associe a une couche
%        sinon le premier element est pris pour toutes les couches
%  parm.res3.cale    parametres pour le calcul de e (voir retchamp) par defaut [1:6] si [] pas calcul du champ (o seulement) 
%  parm.res3.calo    parametres pour le calcul de o (voir retchamp) par defaut i   (o=indice)
%  parm.res3.gauss   0:points regulierement espaces   1:methode de gauss     defaut 0
%  parm.res3.trace   1:trace automatique des champs et de l'objet   defaut 0   
%  parm.res3.champs   champs a tracer et objet (defaut [1,2,3,4,5,6,0])   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<1;dim=2;end;
if dim==2;% 2 D
sym=struct('x',[],'y',[],'pol',0);
parm1=struct('angles',1,'trace',0,'xlimite',[],'ylimite',[],'nx',100,'ny',100,'calcul',1,'champ',0,'ftemp',1,'fperm',[],'sog',1,'li',1);
else;% 1 D
sym=struct('x',[]);
parm1=struct('trace',0,'xlimite',[],'nx',1000,'calcul',1,'champ',1,'ftemp',0,'fperm',[],'sog',1);

end;
parm2=struct('cals',1,'cale',1,'calef',1,'tolh',1.e-6,'tolb',1.e-6,'retss',3,'retgg',0,'result',1);
parm3=struct('npts',10,'cale',[1:6],'calo',i,'sens',1,'caltab',0,'gauss',0,'gauss_x',10,'gauss_y',nan,'trace',0,'champs',[1:6,0],'apod_champ',0);
parm=struct('dim',dim,'sym',sym,'res1',parm1,'res2',parm2,'res3',parm3);





