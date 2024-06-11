function y=retcompare(a,b,k);
%  y=retcompare(a,b,k);  
%  difference entre les elements non infinis et non nan de a et b
%  si k=0 (ou [] ou abs] y=sqrt(2*sum(abs(a-b))^2/(sum(abs(a)^2)+sum(abs(b)^2)))
%  si k=1 y=sqrt(sum(abs(a-b))^2)
%  si k=2 y=max(abs(a-b))
%
%  cas particuliers  y=i  a=[], b non vide
%                    y=2i  a et b n'ont pas le même nombre d'éléments
%                    y=3i  pas d'élément commun fini dans a et b
%                   imag(y)=-1  existance d'elements non finis communs dans a et b
%                   imag(y)=-2  existance d'elements non finis non communs a  a et b
%
%  si 1 seul argument: y=retcompare(a) transforme un cell array a  en vecteur de ses valeurs numeriques y
%
%  s'il y a des fichiers, ils sont lus 
% See also:ISEQUAL
if nargin<2;y=retev(a);return;end;
if nargin<3;k=[];end;if isempty(k);k=0;end;

if isempty(a);if isempty(b);y=0;else;y=i;end;return;end;
a=retev(a);b=retev(b);if length(a)~=length(b);y=2i;return;end;
[f,ff]=retfind(isfinite(a)&isfinite(b));if isempty(f);y=3i;return;end;
%if ~isempty(ff);if isempty(isfinite(a(ff)))&isempty(isfinite(b(ff)));yi=-2i;else yi=-i;end;else;yi=0;end;
if ~isempty(ff);if ~isempty(find(isfinite(a(ff))|isfinite(b(ff))));yi=-2i;else yi=-i;end;else;yi=0;end;
a=a(f);b=b(f);

if k==0;y=sum(abs(a-b).^2);yy=sum(abs(a).^2)+sum(abs(b).^2);if yy~=0;y=sqrt(2*y/yy);end;end;
if k==1;y=sqrt(sum(abs(a-b).^2));end;
if k==2;y=max(abs(a-b));end;
y=y+yi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function aa=retev(a);
% function aa=retev(a);
% transforme  transforme un cell array  en vecteur de ses valeurs numeriques 
if isnumeric(a);aa=a(:).';return;end;

if iscell(a);aa=[];for ii=1:numel(a);aa=[aa,retev(a{ii})];end;return;
else;% pas cell
if ~isempty(a);a=retio(a);% on lit a mais on peut tomber sur un cell_array
if iscell(a);aa=[];for ii=1:numel(a);aa=[aa,retev(a{ii})];end;return;end;
%if ~ischar(a(1));aa=reshape(a,1,numel(a));else aa=[];end;
if ~ischar(a(1));aa=a(:).';else aa=[];end;
else aa=[];end;
end;
if isstruct(aa);aa=[];end;









