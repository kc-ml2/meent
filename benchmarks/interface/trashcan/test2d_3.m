function [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, e] = test2d_3();

    warning('off', 'Octave:possible-matlab-short-circuit-operator');
    warning('off', 'Invalid UTF-8 byte sequences have been replaced.');
    warning('off', 'findstr is obsolete; use strfind instead');

    factor = 1;
    pol = -1;
    n_top = 1;
    n_bot = 1;
    theta = 20;
    phi = 33;
    nn = [11,11];
    period = [770/factor, 770/factor];
    wavelength = 777/factor;
    thickness = 100/factor;


    a = [period(1)/4, period(2)/2, period(1)/2, period(2), 4, 1];
    tt = {1, a};

    retio;
    textures = cell(1,3);
    textures{1} = n_top;
    textures{2} = tt;
    textures{3} = n_bot;

    parm = res0;
    parm.res1.champ = 1;      % calculate precisely
    parm.res1.trace = 1;

    k_parallel = n_top*sind(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

    parm = res0;

    parm.not_io = 1;  % no write data on hard disk
    parm.res1.champ = 1; % the electromagnetic field is calculated accurately
    parm.res3.npts=[11,11,11];

    profile = {[0, thickness, 0], [1, 2, 3]};
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);

    x = linspace(-period(1)/2, period(1)/2, 50);
    y = linspace(-period(2)/2, period(2)/2, 50);
    y = linspace(period(2)/2, -period(2)/2, 50);
    x = linspace(0, period(1), 50);
    y = linspace(period(2), 0, 50);

%    x = [0:1:49] * period(1) / 50 - period(1)/2;
%    x = [1:1:50] * period(1) / 50 - period(1)/2;
%    y = [0:1:49] .* period(2) / 50 - period(2)/2
%    y = [1:1:50] .* period(2) / 50 - period(2)/2
%    y = [50:-1:1] .* period(2) / 50 - period(2)/2
%    y = [49:-1:0] .* period(2) / 50 - period(2)/2

    parm.res3.trace=1;  %trace automatique % automatic trace

    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x,y,aa,profile,einc, parm);

    if pol == 1  % TE
        top_refl_info = res.TEinc_top_reflected;
        top_tran_info = res.TEinc_top_transmitted;
        bottom_refl_info = res.TEinc_bottom_reflected;
        bottom_tran_info = res.TEinc_bottom_transmitted;
    else  % TM
        top_refl_info = res.TMinc_top_reflected;
        top_tran_info = res.TMinc_top_transmitted;
        bottom_refl_info = res.TMinc_bottom_reflected;
        bottom_tran_info = res.TMinc_bottom_transmitted;
    end
    disp(1)
end
