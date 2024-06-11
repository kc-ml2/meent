function  pl=retguide(nm,nd,ld,betay)
% function  guide=retguide(indice_metal,indice_dielectrique,lambda,betay);
% retourne une structure contenant;
%      constante_de_propagation, khi_metal, khi_dielectrique
%      test_d_existence = 1 si le plasmon existe ,0 sinon
% si lambda est precise on a aussi les distances d'attenuation en intensite d'un facteur e
%      distance_de_propagation,penetration_metal,penetration_dielectrique
%      attenuaion en db/mm si lambda est en microns
%      poynting=flux du vecteur de Poynting d'un plasmon dont le vecteur H vaut 1 sur le dioptre
%      int_lorentz: integrale de Lorentz de ce meme plasmon ( poynting voisin de int_lorentz/4 )
% par defaut   indice_dielectrique=1
%
%% EXEMPLE  distance de propagation du plasmon de l'or fonction de lambda
%  figure;ld=linspace(.4,2.5,1000);
%  pl=retplasmon(retindice(ld,2),1,ld);plot(ld,pl.distance_de_propagation);
%  grid;xlabel('lambda microns ');title('distance de propagation du plasmon de l''or (microns)')
%
% See also:RETPL,RETDB


if nargin<4;betay=0;end;
if nargin<2;nd=1;end; % air par defaut
beta=sqrt(1./(nd.^-2+nm.^-2)-betay^2);khim=retsqrt(nm.^2-(beta.^2+betay^2),-1);khid=retsqrt(nd.^2-(beta.^2+betay^2),-1);
test=abs((khim+(nm./nd).^2.*khid)./(2*khim))<1.e-10;
%e=[beta,betay,-i*(beta^2+betay^2)/khim];
pl=struct('constante_de_propagation',beta,'khi_metal',khim,'khi_dielectrique',khid,'test_d_existence',test);
if nargin>2;a=ld/(4*pi);
pl.distance_de_propagation=a./abs(imag(beta));
pl.penetration_metal=a./abs(imag(khim));
pl.penetration_dielectrique=a./abs(imag(khid));
pl.db_par_mm=retdb(beta,ld);
k0=2*pi./ld;
pl.poynting=real((beta.^2+betay^2)./(4*k0.*nm.^2.*beta))./imag(khim)+real((beta.^2+betay^2)./(4*k0.*nd.^2.*beta))./imag(khid);
pl.int_lorentz=i*((beta.^2+betay^2)./beta).*(1./(k0.*nm.^2.*khim)+1./(k0.*nd.^2.*khid));
end;





