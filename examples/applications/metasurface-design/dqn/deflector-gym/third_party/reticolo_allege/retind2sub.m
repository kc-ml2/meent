function varargout=retind2sub(s,II)
% fonctionnement identique à ind2sub de matlab mais plus rapide ( facteur 1.5)
% See also SUB2IND IND2SUB RETSUB2IND 
II=II-1;
n=length(s);
switch n;
case 3;varargout{1}=mod(II,s(1));II=(II-varargout{1})/s(1);varargout{1}=varargout{1}+1;
	   varargout{2}=mod(II,s(2));varargout{3}=(II-varargout{2})/s(2)+1;varargout{2}=varargout{2}+1;
otherwise;

for ii=1:n-1;
varargout{ii}=mod(II,s(ii));II=(II-varargout{ii})/s(ii);varargout{ii}=varargout{ii}+1;
end;
varargout{n}=(II-varargout{n-1})/s(n-1)+1;
end;
