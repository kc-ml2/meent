function font=retfont(k,type);
%  font=retfont 
%%% EXEMPLE DE TRACE
% figure;
% subplot(3,1,1);hold on;for ii=1:5;plot(rand(1,5),retfont(ii));end;
% subplot(3,1,2);hold on;for ii=1:5;plot(rand(1,5),retfont(ii,2));end;
% subplot(3,1,3);hold on;for ii=1:5;plot(rand(1,5),retfont(ii,3));end;


if nargin<1;font={'interpreter','none','fontname','Times New Roman','fontweigh','bold'};return;end;

if nargin<2;type=1;end;
if abs(k(1)-round(k(1)))>100*eps;% k est le handel d'un plot
end;

% lin={{'-k','-r','-g','-b','-m','-c','--k','--r','--g','--b','--m','--c','-.k','-.r','-.g','-.b','-.m','-.c',':k',':r',':g',':b',':m',':c'};...
% 	{'.k','.r','.g','.b','.m','.c','*k','*r','*g','*b','*m','*c','+k','+r','+g','+b','+m','+c','xk','xr','xg','xb','xm','xc'}};
lin={{'-k',':r','-.g','--b','-m',':c','-.k','--r','-g',':b','-.m','--c','-k',':r','-.g','--b','-m',':c','-.k','--r','-g',':b','-.m','--c'};...
	{'or','xg','+b','*m','sc','dk','vr','^g','<b','>m','pc','hk','og','xb','+m','*c','sk','dr','vg','^b','<m','>c'};...
	{'.-k','o:r','x-.g','+--b','*-m','s:c','d-.k','v--r','^-g','<:b','>-.m','p--c','h-k','.:r','o-.g','x--b','*-m','s:c','d-.k','v--r','^-g','<:b','>-.m','p--c'}};
	
font=lin{type}{mod(k,length(lin{type})-1)+1};


