function varargout=reteig(varargin)
%%%%[v,d]=reteig(varargin); idem a eig mais marche avec les matrices sparses
% en les tranformant en matrices pleines
% if nargout<2;v=eig(varargin{:});return;end;
% for ii=1:nargin;varargin{ii}=full(varargin{ii});end;[v,d]=eig(varargin{:});
% les tests ont montré que l'option 'nobalance' était plus stable et plus rapide



for ii=1:nargin;varargin{ii}=full(varargin{ii});end;

if nargin<2;

% 	if nargin==1;n=size(varargin{1},1);
% 	if mod(n,2)==0;n=n/2;
% 	a=max(abs(varargin{1}(:)));
% 	if max(max(abs(varargin{1}(1:n,n+1:2*n))))<1.e-12*a;[varargout{1:nargout}]=reteigg(varargin{1},n,1);return;end;
% 	if max(max(abs(varargin{1}(n+1:2*n,1:n))))<1.e-12*a;[varargout{1:nargout}]=reteigg(varargin{1},n,-1);return;end;
% 	end;
% 	end;

[varargout{1:nargout}]=eig(varargin{:},'nobalance');
else;
[varargout{1:nargout}]=eig(varargin{:});
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v,d]=reteigg(a,n,sens),sens
if nargout<2;v=[eig(a(1:n,1:n),'nobalance');eig(a(n+1:2*n,n+1:2*n),'nobalance')];return;end;
[v1,d1]=eig(a(1:n,1:n),'nobalance');[v2,d2]=eig(a(n+1:2*n,n+1:2*n),'nobalance');d=diag([diag(d1);diag(d2)]);
if sens>0;% a12=0
w=((v2\a(n+1:2*n,1:n))*v1)./(repmat(diag(d1).',n,1)-repmat(diag(d2),1,n));v=[v1,zeros(n);v2*w,v2];
else;    % a21=0
w=((v1\a(1:n,n+1:2*n))*v2)./(repmat(diag(d2).',n,1)-repmat(diag(d1),1,n));v=[v1,v1*w;zeros(n),v2];
end;

