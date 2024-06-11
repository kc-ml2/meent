function [ss,s]=retcrs(ss,s);
% croissance des matrices pour rendre possible de produit ss par s
% (produit 'intelligent' utile pour les sources)

if size(s,1)<6; %matrices S
n1=s{2};n2=size(s{1},2)-n1;n3=s{3};n4=size(s{1},1)-n3;
nn1=ss{2};nn2=size(ss{1},2)-nn1;nn3=ss{3};nn4=size(ss{1},1)-nn3;
if nn1>n3;n=nn1-n3;% croissance de S
s{1}=[[s{1}(1:n3,1:n1),zeros(n3,n),s{1}(1:n3,n1+1:n1+n2)];[zeros(n,n1),eye(n),zeros(n,n2)];[s{1}(n3+1:n3+n4,1:n1),zeros(n4,n),s{1}(n3+1:n3+n4,n1+1:n1+n2)]];
n1=n1+n;n3=n3+n;s{2}=n1;s{3}=n3;
end;    
if nn4>n2;n=nn4-n2;
s{1}=[[s{1},zeros(n3+n4,n)];[zeros(n,n1+n2),eye(n)]];    
n2=n2+n;n4=n4+n;    
end;    
if n3>nn1;n=n3-nn1; % croissance de SS
ss{1}=[[ss{1}(1:nn3,1:nn1),zeros(nn3,n),ss{1}(1:nn3,nn1+1:nn1+nn2)];[zeros(n,nn1),eye(n),zeros(n,nn2)];[ss{1}(nn3+1:nn3+nn4,1:nn1),zeros(nn4,n),ss{1}(nn3+1:nn3+nn4,nn1+1:nn1+nn2)]];    
nn1=nn1+n;nn3=nn3+n;ss{2}=nn1;ss{3}=nn3;    
end;    
if n2>nn4;n=n2-nn4;
ss{1}=[[ss{1},zeros(nn3+nn4,n)];[zeros(n,nn1+nn2),eye(n)]];
nn2=nn2+n;nn4=nn4+n;
end;    

else      %matrices G
n1=s{3};n2=s{4};n3=s{5};n4=s{6};m=size(s{1},1);nn3=ss{5};nn4=ss{6};nn1=ss{3};nn2=ss{4};mm=size(ss{1},1); 
if nn1>n3;n=nn1-n3;   % croissance de S
s{1}=[[s{1}(:,1:n1),zeros(m,n),s{1}(:,n1+1:n1+n2)];[zeros(n,n1),eye(n),zeros(n,n2)]];    
s{2}=[[s{2}(:,1:n3),zeros(m,n),s{2}(:,n3+1:n3+n4)];[zeros(n,n3),eye(n),zeros(n,n4)]];    
n1=n1+n;n3=n3+n;s{3}=n1;s{5}=n3;m=m+n;    
end    
if nn2>n4;n=nn2-n4; 
s{1}=[[s{1},zeros(m,n)];[zeros(n,n1+n2),eye(n)]];    
s{2}=[[s{2},zeros(m,n)];[zeros(n,n3+n4),eye(n)]];    
n2=n2+n;n4=n4+n;s{4}=n2;s{6}=n4;m=m+n;    
end    
if n3>nn1;n=n3-nn1; % croissance de SS
ss{1}=[[ss{1}(:,1:nn1),zeros(mm,n),ss{1}(:,nn1+1:nn1+nn2)];[zeros(n,nn1),eye(n),zeros(n,nn2)]];    
ss{2}=[[ss{2}(:,1:nn3),zeros(mm,n),ss{2}(:,nn3+1:nn3+nn4)];[zeros(n,nn3),eye(n),zeros(n,nn4)]];    
nn1=nn1+n;nn3=nn3+n;ss{3}=nn1;ss{5}=nn3;mm=mm+n;    
end    
if n4>nn2;n=n4-nn2;
ss{1}=[[ss{1},zeros(mm,n)];[zeros(n,nn1+nn2),eye(n)]];    
ss{2}=[[ss{2},zeros(mm,n)];[zeros(n,nn3+nn4),eye(n)]];    
nn2=nn2+n;nn4=nn4+n;ss{4}=nn2;ss{6}=nn4;mm=mm+n;    
end    

end    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    