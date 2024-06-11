function [betaa,masque]=retapod(betaa,apod);% apodisation  
%function [betaa,masque]=retapod(betaa,apod);% apodisation  
% multiplie les lignes de betaa par une fonction d'apodisation fonction des colonnes (masque vecteur ligne)
%  betaa=diag(masque)*betaa
%
%   apod:parametre
%                0 pas d'apodisation  1 hanning  2  hamming  3  flat top  4  blackman   5  blackman harris
%                6.12..  trapeze (les decimales representent un parametre)   7.32..  trapeze arrondi
%                8  schwartz exp(-1/(1-x^2))  9  exp(-1/(1-x^4)) 9.12 exp(-1/(1-x^(1/.12))) 
%               11 tf d'un arche de cos peut etre utile pour les sources..a tester
%     si apod a une partie imaginaire,troncature dans la proportion de imag(apod) (<1) puis apodisation avec real(apod)
%
%  retapod(apod):trace de la fonction d'apodisation
% 
% 
%% Tracé de la decroissance de la TF des fonctions d'apodisation
% figure;hold on;leg=cell(1,0);
% for apod=[1:5,7.25,8,9];leg=[leg,{num2str(apod)}];
% N=128;a=[zeros(1,2*N),retapod(ones(N+1,1),apod).',zeros(1,2*N)];a=a(1:end-1);semilogy(1:length(a),abs(ifftshift(fft(fftshift(a)))/N),retfont(round(apod)));
% set(gca,'yscale','log');legend(leg);end



if nargin<2;retchamp(betaa);return;end; %trace de la fonction d'apodisation ( pour tests )
mm=size(betaa,1);masque=ones(1,mm);
tronc=imag(apod);apod=real(apod);if tronc~=0; % troncature
nt=floor(mm*tronc/2);beta=betaa((1+nt):(mm-nt),:);betaa(:)=0;masque(:)=0;
[betaa((1+nt):(mm-nt),:),masque((1+nt):(mm-nt))]=retapod(beta,apod);
return;end;

b=abs(apod-fix(apod));if b==0;b=.25;end;apod=fix(apod);% parametres exemple apod=-1.1  -> b=.1  apod=-1
switch apod;
case 1;a=[.5,.5,0];% hanning
case 2;a=[.54,.46,0];% hamming
case 3;a=[.2810639,.5208972,-0.1980399];% flat top
case 4;a=[.42659071,.49656062,-0.07684867];% blackman 
case 5;a=[.42323,.49755,-0.07922];% blackman harris
case 6;mmm=ceil(mm*b);masque=[1:mmm,mmm*ones(1,mm-mmm)];masque=masque.*fliplr(masque);% trapeze
case 7;mmm=ceil(mm*b);x=[1:mmm]/mmm;masque=[(3-2*x).*x.^2,ones(1,mm-mmm)];masque=masque.*fliplr(masque);% trapeze arrondi
case 8;mmm=(mm+1)/2;masque=exp(-1./(1-(([1:mm]-mmm)/mmm).^2));% schwartz exp(-1/(1-x^2))
case 9;mmm=(mm+1)/2;masque=exp(-1./(1-abs(([1:mm]-mmm)/mmm).^(1/b)));% exp(-1/(1-x^(1/b))) 9.25  exp(-1/(1-x^4))
case 10;a=[.35875,.48829,-.14128,.01168];masque=a(1)+a(2)*sin(2*pi*([1:mm]/(mm+1)-.25))+a(3)*sin(4*pi*([1:mm]/(mm+1)-.125))+a(4)*sin(6*pi*([1:mm]/(mm+1)-.25/3));% blackman harris 4 termes
case 11;x=linspace(-.5,.5,mm)/b;masque=(retsinc(x)+.5*retsinc(x-1)+.5*retsinc(x+1));% cos support borné
case 12;x=linspace(-.5,.5,mm)/b;masque=(retsinc(x)+.5*retsinc(x-1)+.5*retsinc(x+1)).^2;% 11^2
case 13;x=linspace(-1,1,mm)/b;masque=((6*retsinc(x)+4*retsinc(x-1)+4*retsinc(x+1)+retsinc(x-2)+retsinc(x+2))/16);% cos^2
case 14;x=linspace(-1,1,mm)/b;masque=exp(-(x/2).^2);% gaussienne
end;
if ismember(apod,1:5);masque=a(1)+a(2)*sin(2*pi*([1:mm]/(mm+1)-.25))+a(3)*sin(4*pi*([1:mm]/(mm+1)-.125));end;
masque=masque/max(masque);betaa=full(retdiag(masque)*betaa);
