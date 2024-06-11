function [x,y]=retginput(n);
% comme ginput mais sans bugg ...
if nargin<1;n=1;end;x=zeros(n,1);y=zeros(n,1);
figure(gcf);
try
set(gcf,'Pointer','fullcrosshair');
for ii=1:n;
waitforbuttonpress;
xy=get(gca,'CurrentPoint');
x(ii)=xy(1,1);y(ii)=xy(1,2);
end;
set(gcf,'Pointer','arrow');
catch;
set(gcf,'Pointer','arrow');
end;
