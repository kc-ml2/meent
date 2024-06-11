function c=retscale(varargin);
%  function c=retscale(a,b   d); c=sum(a.*b...  .*d)  
for ii=2:length(varargin);varargin{1}(:)=varargin{1}(:).*varargin{ii}(:);end;
c=sum(varargin{1}(:));
