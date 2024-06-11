function x=subsref(b,s); %Version 6
if retversion>60; x=subsref7(b,s);return;end;% MATLAB 7

a=getfield(struct(b),'ret');

champ1=s(1).subs;
if length(s)==1;x=eval(['a.',champ1]);return;end;% appel sans indices 
champ2=s(2).subs;
if length(s)==2;x=eval(['a.',champ1,'.',champ2]);return;end;% appel sans indices 




if champ1(1)=='J';% matrice de Jones
ii=s(3).subs{1};if length(s(3).subs)==2;ii=ii(1)+i*s(3).subs{2};end;% transformation en notation complexe

if ismember(champ2,fieldnames(a));% 1 D
champ1=champ2;aa=eval(['a.',champ1]);
if size(aa.order,2)==2;order=aa.order(:,1)+i*aa.order(:,2);else;order=aa.order;end;
num=find(order==ii);
if ~isempty(num);x=aa.amplitude(num);else;x=0;end;
else;% 2d;

x=zeros(2);
champ1=['TE',champ2];aa=eval(['a.',champ1]);
if size(aa.order,2)==2;order=aa.order(:,1)+i*aa.order(:,2);else;order=aa.order;end;
num=find(order==ii);
if ~isempty(num);
x(1,1)=aa.amplitude_TE(num);
x(2,1)=aa.amplitude_TM(num);
end
champ1=['TM',champ2];aa=eval(['a.',champ1]);
if size(aa.order,2)==2;order=aa.order(:,1)+i*aa.order(:,2);else;order=aa.order;end;
num=find(order==ii);
if ~isempty(num);
x(1,2)=aa.amplitude_TE(num);
x(2,2)=aa.amplitude_TM(num);
end;
end;% 1D 2 D
return;end;

aa=eval(['a.',champ1]);







switch s(3).type;
case '{}';
ii=s(3).subs{1};if length(s(3).subs)==2;ii=ii(1)+i*s(3).subs{2};end;% transformation en notation complexe
if size(aa.order,2)==2;order=aa.order(:,1)+i*aa.order(:,2);else;order=aa.order;end;
num=find(order==ii);
if ~isempty(num);
x=eval(['aa.',champ2]);x=x(num,:);
else;
switch(champ2);case {'efficiency','efficiency_TE','efficiency_TM','amplitude','amplitude_TE','amplitude_TM'};x=0;
otherwise;x=zeros(0,size(eval(['aa.',champ2]),2));end;
end;
case '()';x=eval(['aa.',champ2]);x=x(s(3).subs{:});% appel normal
end

