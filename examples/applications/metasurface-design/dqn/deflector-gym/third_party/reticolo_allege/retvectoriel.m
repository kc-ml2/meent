function v=retvectoriel(v,vv);
% v=retvectorial(v,vv); v=v vectorial vv 
% size(v)=[n,3]  size(vv)=[n,3]
v=[v(:,2).*vv(:,3)-vv(:,2).*v(:,3),v(:,3).*vv(:,1)-vv(:,3).*v(:,1),v(:,1).*vv(:,2)-vv(:,1).*v(:,2)];



