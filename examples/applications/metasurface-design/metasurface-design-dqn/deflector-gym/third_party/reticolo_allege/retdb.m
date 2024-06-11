function db=retdb(n,ld);
% db=retdb(n,ld);
% attenuation en db/mm d'un indice complexe n à la longueur d'onde ld (microns)
% 
% conversion ld MICRONS , en GIGHA HERTZ  ou eV
% Gh=retdb(ld,'ld_2_Gh');(ou Gh=retdb(ld);)
% eV=retdb(ld,'ld_2_eV');
% ld=retdb(eV,'eV_2_ld');
% ld=retdb(Gh,'Gh_2_ld');
% db=retdb(attenuation_en_energie,'attenuation_en_energie_2_db');
% attenuation_en energie=retdb(db,'db_2_attenuation_en energie');


% Definition des db  db=10*log10(Energie2/Energie1)




if nargin<2;db=2.99792458e5./n;return;end;
q=1.6022E-19;h=6.6261E-34;c=299792458;
switch ld;
case {'ld_2_Gh','Gh_2_ld'};db=(1.e-3*c)./n;
case {'ld_2_eV','eV_2_ld'};db=(h*c/(1.e-6*q))./n;
case 'Gh_2_eV';db=(h/(1.e-9*q)).*n;
case 'eV_2_Gh';db=(1.e-9*q/h).*n;
case 'attenuation_en_energie_2_db';db=10*log10(n);
case 'db_2_attenuation_en energie';db=10.^(n/10);
otherwise;db=imag(n).*(40000*pi./(log(10)*ld));
end;