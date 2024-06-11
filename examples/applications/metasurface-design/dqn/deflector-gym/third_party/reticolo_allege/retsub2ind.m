function a=retsub2ind(s,varargin)
% fonctionnement identique à sub2ind de matlab mais plus rapide ( facteur 3)
% See also SUB2IND IND2SUB RETIND2SUB 

n=length(s);
switch n;
case 1;a=varargin{1};
case 2;a=varargin{1}+s(1)*(varargin{2}-1);
case 3;a=varargin{1}+s(1)*(varargin{2}-1)+(s(1)*s(2))*(varargin{3}-1);
case 4;a=varargin{1}+s(1)*(varargin{2}-1)+(s(1)*s(2))*(varargin{3}-1)+(s(1)*s(2)*s(3))*(varargin{4}-1);
otherwise;
varargin{n}=varargin{n}-1;	
for ii=n-1:-1:1;varargin{n}=varargin{ii}-1+s(ii)*varargin{n};varargin{ii}=[];end;
a=varargin{n}+1;
end;
