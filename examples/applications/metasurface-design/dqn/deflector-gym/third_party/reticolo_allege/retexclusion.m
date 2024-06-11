function L=retexclusion(sommets,centre);
if nargin<2;centre=[0,0];end;
sommets(:,1)=sommets(:,1)-centre(1);
sommets(:,2)=sommets(:,2)-centre(2);

n=size(sommets,1);
sommets=[sommets;sommets(1,:)];
t=unwrap(atan2(sommets(:,2),sommets(:,1)));
t0=atan2(sommets(1:n,1)-sommets(2:n+1,1),-sommets(1:n,2)+sommets(2:n+1,2));
L=zeros(2);


L22=sum(.5*cos(2*t0).*(t(2:n+1)-t(1:n))-.25*(sin(2*t(2:n+1))-sin(2*t(1:n)))+.5*sin(2*t0).*log(cos(t(2:n+1)-t0)./cos(t(1:n)-t0)));

L12=sum(-.5*sin(2*t0).*(t(2:n+1)-t(1:n))-.25*(cos(2*t(2:n+1))-cos(2*t(1:n)))+.5*cos(2*t0).*log(cos(t(2:n+1)-t0)./cos(t(1:n)-t0)));

L=[[-L22,L12];[L12,L22]]/(2*pi)+.5*eye(2);
