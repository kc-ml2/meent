function retsize(varargin);
% size d'un ensemble d'objets
% retsize(A,B,...)
for ii=1:length(varargin);sz=size(varargin{ii});disp(['size(',inputname(ii),')=[ ',int2str(sz),']']);end;	