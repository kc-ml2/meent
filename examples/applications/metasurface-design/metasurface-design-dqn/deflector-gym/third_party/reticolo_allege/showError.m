figure;
rect = reshape(outputs,11,21)-efficiency;
pcolor(xx,yy,rect*100);
%colormap('hot')
colorbar()

title([num2str(angle),'-',num2str(wavelength),'-',num2str(efficiency)])
xlabel('x order');
ylabel('y order');