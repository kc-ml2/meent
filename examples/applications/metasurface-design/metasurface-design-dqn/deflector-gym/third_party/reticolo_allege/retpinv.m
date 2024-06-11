function p=retpinv(p);p=inv(p);return;
if issparse(p);p=inv(p);return;end;
if  rcond(p)<1.e-10;p=pinv(p);else p=inv(p);end; % le test doit etre fait dans ce sens pour le cas de nan
