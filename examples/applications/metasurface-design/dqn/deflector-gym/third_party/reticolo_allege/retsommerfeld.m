function [e,o]=retsommerfeld(n,pol,arete,teta_plan,teta_incident,x,y)
% function [e,o]=retsommerfeld(n,pol,arete,teta_plan,teta_incident,x,y);
% diffraction du demi plan de sommerfeld  partant du point [arete(1),arete(2)] de direction: teta_plan
% onde plane incidente de direction teta_incident ( champ egal à 1 en [0,0])
%  pol=2;x=linspace(-10,10,150);y=linspace(-10,10,150);
%%EXEMPLE
%  pol=2;x=linspace(-10,10,150);y=linspace(-10,10,150);
%  [e,o]=retsommerfeld(1.5,pol,[-1,-2],175*pi/180,10*pi/180,x,y);rettchamp(e,o,x,y,pol,[1:3,i]);
%
% See also:RETCHAMP RETPOINT


nx=length(x);ny=length(y);
teta_plan=teta_plan-pi;% dans le programme teta_plan correspond à la partie non metallique
co=cos(teta_plan);si=sin(teta_plan);
[Y,X]=ndgrid(y-arete(2),x-arete(1));
[X,Y]=deal(X*co+Y*si,-X*si+Y*co);
teta0=mod(teta_incident-teta_plan+pi,2*pi)-pi;% determination entre -pi et pi
teta=mod(atan2(Y,X)+pi,2*pi)-pi;r=sqrt(X.^2+Y.^2);clear X Y;
E1=exp(-i*n*r.*cos(teta-teta0)).*(.5+(1/(1+i))*retfresnel(2*sqrt((n/pi)*r).*cos((teta-teta0)/2)));
Hn1=i*n*cos(teta-teta0).*E1-exp(i*n*r).*(1/(1+i)).*sqrt(n./(pi*r)).*cos((teta-teta0)/2);
Ht1=i*n*sin(teta-teta0).*E1-exp(i*n*r).*(1/(1+i)).*sqrt(n./(pi*r)).*sin((teta-teta0)/2);
teta0=2*pi-teta0;
E2=exp(-i*n*r.*cos(teta-teta0)).*(.5+(1/(1+i))*retfresnel(2*sqrt((n/pi)*r).*cos((teta-teta0)/2)));
Hn2=i*n*cos(teta-teta0).*E2-exp(i*n*r).*(1/(1+i)).*sqrt(n./(pi*r)).*cos((teta-teta0)/2);
Ht2=i*n*sin(teta-teta0).*E2-exp(i*n*r).*(1/(1+i)).*sqrt(n./(pi*r)).*sin((teta-teta0)/2);
e=zeros(ny,nx,3);

switch pol;
case 0;e(:,:,1)=E1-E2;Ht=Ht1-Ht2;Hn=Hn1-Hn2;
case 2;e(:,:,1)=E1+E2;Ht=(Ht1+Ht2)/(n^2);Hn=(Hn1+Hn2)/(n^2);
end;
co=cos(teta+teta_plan);
si=sin(teta+teta_plan);
e(:,:,2)=Ht.*co-Hn.*si;
e(:,:,3)=Ht.*si+Hn.*co;

if nargout>1;% on remplit o et on complete e
o=ones(ny,nx,3);
switch pol;
case 0;o(:,:,1)=n^2;
case 2;o(:,:,2)=n^2;o(:,:,3)=n^2;
end;
e=cat(3,e,e(:,:,2).*o(:,:,2));
end;
e(:,:,2:end)=-i*e(:,:,2:end);% declonage

%ret;pol=2;x=linspace(240,260,50);y=linspace(-10,10,60);e=retsommerfeld(1,pol,[0,0],0*pi/180,90*pi/180,x,y);rettchamp(e,[],x,y,pol,[1:3,-1])
%rettchamp(e,[],x,y,pol,-1)