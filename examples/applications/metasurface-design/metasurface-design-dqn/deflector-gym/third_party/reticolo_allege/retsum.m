function c=retsum(a,b);
%  function c=retsum(a,b);
%  c=a+b  si a=[] ou b=[] creation de la matrice c
if isempty(a);c=b;return;end;
if isempty(b);c=a;return;end;
c=a+b;