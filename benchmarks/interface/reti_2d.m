
function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d(ex_case);
    warning('off', 'Octave:possible-matlab-short-circuit-operator');
    warning('off', 'Invalid UTF-8 byte sequences have been replaced.');
    warning('off', 'findstr is obsolete; use strfind instead');

    if ex_case == 1
        [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_1();
    elseif ex_case == 2
        [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_2();
    elseif ex_case == 3
        [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_3();
    elseif ex_case == 4
        [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_4();
    elseif ex_case == 5
        [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_5();
    elseif ex_case == 6
        [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_6();
    elseif ex_case == 7
        [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_7();
    end

end

function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_1();

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

    b = [-period(1)/4, -period(2)/4, period(1)/2, period(2)*5/10, 3, 1];
    c = [-period(1)/10*4, period(2)/10*4, period(1)*2/10, period(2)*2/10, 4, 1];
    d = [-period(1)/10*2, period(2)/10*4, period(1)*2/10, period(2)*2/10, 6, 1];

    b = [-period(1)/4+period(1)/2, -period(2)/4+period(2)/2, period(1)/2, period(2)*5/10, 3, 1];
    c = [-period(1)/10*4+period(1)/2, period(2)/10*4+period(2)/2, period(1)*2/10, period(2)*2/10, 4, 1];
    d = [-period(1)/10*2+period(1)/2, period(2)/10*4+period(2)/2, period(1)*2/10, period(2)*2/10, 6, 1];

    tt = {1, b, c, d};

    retio;
    textures = cell(1,3);
    textures{1} = n_top;
    textures{2} = tt;
    textures{3} = n_bot;

    parm = res0;
    parm.res1.champ = 1;      % calculate precisely
%    parm.res1.trace = 1;
%    parm.res3.trace = 1;  % trace automatique % automatic trace

    k_parallel = n_top*sind(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

    parm = res0;

    parm.not_io = 1;  % no write data on hard disk
    parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%    parm.res3.npts=[0,1,0];
    parm.res3.npts=[11,11,11];

    profile = {[0, thickness, 0], [1, 2, 3]};
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);

    x = linspace(-period(1)/2, period(1)/2, 11);
    y = linspace(period(2)/2, -period(2)/2, 11);
    x = linspace(0, period(1), 11);
    y = linspace(period(2), 0, 11);

%    x = [0:1:49] * period(1) / 50 - period(1)/2;
%    x = [1:1:50] * period(1) / 50 - period(1)/2;
%    y = [0:1:49] .* period(2) / 50 - period(2)/2
%    y = [1:1:50] .* period(2) / 50 - period(2)/2
%    y = [50:-1:1] .* period(2) / 50 - period(2)/2
%    y = [49:-1:0] .* period(2) / 50 - period(2)/2


    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x,y,aa,profile,einc, parm);

%    if pol == 1  % TE
%        top_refl_info = res.TEinc_top_reflected;
%        top_tran_info = res.TEinc_top_transmitted;
%        bottom_refl_info = res.TEinc_bottom_reflected;
%        bottom_tran_info = res.TEinc_bottom_transmitted;
%    else  % TM
%        top_refl_info = res.TMinc_top_reflected;
%        top_tran_info = res.TMinc_top_transmitted;
%        bottom_refl_info = res.TMinc_bottom_reflected;
%        bottom_tran_info = res.TMinc_bottom_transmitted;
%    end

    top_refl_info_te = res.TEinc_top_reflected;
    top_tran_info_te = res.TEinc_top_transmitted;
    top_refl_info_tm = res.TMinc_top_reflected;
    top_tran_info_tm = res.TMinc_top_transmitted;

    bottom_refl_info_te = res.TEinc_bottom_reflected;
    bottom_tran_info_te = res.TEinc_bottom_transmitted;
    bottom_refl_info_tm = res.TMinc_bottom_reflected;
    bottom_tran_info_tm = res.TMinc_bottom_transmitted;

end

function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_2();

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

    b = [-period(1)/4, -period(2)/4, period(1)/2, period(2)*5/10, 3, 1];
    b = [period(1)/4, period(2)/4, period(1)/2, period(2)*5/10, 3, 1];

    tt = {1, b};

    retio;
    textures = cell(1,3);
    textures{1} = n_top;
    textures{2} = tt;
    textures{3} = n_bot;

    parm = res0;
    parm.res1.champ = 1;      % calculate precisely
%    parm.res1.trace = 1;
%    parm.res3.trace = 1;  % trace automatique % automatic trace

    k_parallel = n_top*sind(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

    parm = res0;

    parm.not_io = 1;  % no write data on hard disk
    parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%    parm.res3.npts=[0,1,0];
    parm.res3.npts=[11,11,11];

    profile = {[0, thickness, 0], [1, 2, 3]};
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);

    x = linspace(-period(1)/2, period(1)/2, 11);
    y = linspace(-period(2)/2, period(2)/2, 11);
    y = linspace(period(2)/2, -period(2)/2, 11);
    x = linspace(0, period(1), 11);
    y = linspace(period(2), 0, 11);

%    x = [0:1:49] * period(1) / 50 - period(1)/2;
%    x = [1:1:50] * period(1) / 50 - period(1)/2;
%    y = [0:1:49] .* period(2) / 50 - period(2)/2
%    y = [1:1:50] .* period(2) / 50 - period(2)/2
%    y = [50:-1:1] .* period(2) / 50 - period(2)/2
%    y = [49:-1:0] .* period(2) / 50 - period(2)/2

    parm.res3.trace=0;  %trace automatique % automatic trace

    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x,y,aa,profile,einc, parm);

%    if pol == 1  % TE
%        top_refl_info = res.TEinc_top_reflected;
%        top_tran_info = res.TEinc_top_transmitted;
%        bottom_refl_info = res.TEinc_bottom_reflected;
%        bottom_tran_info = res.TEinc_bottom_transmitted;
%    else  % TM
%        top_refl_info = res.TMinc_top_reflected;
%        top_tran_info = res.TMinc_top_transmitted;
%        bottom_refl_info = res.TMinc_bottom_reflected;
%        bottom_tran_info = res.TMinc_bottom_transmitted;
%    end

    top_refl_info_te = res.TEinc_top_reflected;
    top_tran_info_te = res.TEinc_top_transmitted;
    top_refl_info_tm = res.TMinc_top_reflected;
    top_tran_info_tm = res.TMinc_top_transmitted;

    bottom_refl_info_te = res.TEinc_bottom_reflected;
    bottom_tran_info_te = res.TEinc_bottom_transmitted;
    bottom_refl_info_tm = res.TMinc_bottom_reflected;
    bottom_tran_info_tm = res.TMinc_bottom_transmitted;
end


function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_3();

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
    parm.res1.trace = 0;

    k_parallel = n_top*sind(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

    parm = res0;

    parm.not_io = 1;  % no write data on hard disk
    parm.res1.champ = 1; % the electromagnetic field is calculated accurately
    parm.res3.npts=[11,11,11];

    profile = {[0, thickness, 0], [1, 2, 3]};
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);

    x = linspace(-period(1)/2, period(1)/2, 11);
    y = linspace(-period(2)/2, period(2)/2, 11);
    y = linspace(period(2)/2, -period(2)/2, 11);
    x = linspace(0, period(1), 11);
    y = linspace(period(2), 0, 11);

%    x = [0:1:49] * period(1) / 50 - period(1)/2;
%    x = [1:1:50] * period(1) / 50 - period(1)/2;
%    y = [0:1:49] .* period(2) / 50 - period(2)/2
%    y = [1:1:50] .* period(2) / 50 - period(2)/2
%    y = [50:-1:1] .* period(2) / 50 - period(2)/2
%    y = [49:-1:0] .* period(2) / 50 - period(2)/2

    parm.res3.trace=0;  %trace automatique % automatic trace

    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x,y,aa,profile,einc, parm);

%    if pol == 1  % TE
%        top_refl_info = res.TEinc_top_reflected;
%        top_tran_info = res.TEinc_top_transmitted;
%        bottom_refl_info = res.TEinc_bottom_reflected;
%        bottom_tran_info = res.TEinc_bottom_transmitted;
%    else  % TM
%        top_refl_info = res.TMinc_top_reflected;
%        top_tran_info = res.TMinc_top_transmitted;
%        bottom_refl_info = res.TMinc_bottom_reflected;
%        bottom_tran_info = res.TMinc_bottom_transmitted;
%    end

    top_refl_info_te = res.TEinc_top_reflected;
    top_tran_info_te = res.TEinc_top_transmitted;
    top_refl_info_tm = res.TMinc_top_reflected;
    top_tran_info_tm = res.TMinc_top_transmitted;

    bottom_refl_info_te = res.TEinc_bottom_reflected;
    bottom_tran_info_te = res.TEinc_bottom_transmitted;
    bottom_refl_info_tm = res.TMinc_bottom_reflected;
    bottom_tran_info_tm = res.TMinc_bottom_transmitted;
end


function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_4();

    warning('off', 'Octave:possible-matlab-short-circuit-operator');
    warning('off', 'Invalid UTF-8 byte sequences have been replaced.');
    warning('off', 'findstr is obsolete; use strfind instead');

    factor = 1;
    pol = 1;
    n_top = 1;
    n_bot = 1;
    theta = 0;
    phi = 0;
    nn = [11,11];
    period = [480/factor, 480/factor];
    wavelength = 550/factor;
    thickness = 220/factor;


    a = [0+240, 120+240, 160, 80, 4, 1];
    b  = [0+240, -120+240, 160, 80, 4, 1];
    c = [120+240, 0+240, 80, 160, 4, 1];
    d = [-120+240, 0+240, 80, 160, 4, 1];

    textures = cell(1,3);
    textures{1} = n_top;
    textures{2} = {1,a, b, c, d};
    textures{3} = n_bot;


    parm = res0;
    parm.res1.champ = 1;      % calculate precisely
    parm.res1.trace = 0;

    k_parallel = n_top*sind(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

    parm = res0;

    parm.not_io = 1;  % no write data on hard disk
    parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%    parm.res3.npts=[0,1,0];
    parm.res3.npts=[11,11,11];

    %parm.res1.trace = 1; % show the texture
    %
    %textures = cell(1, size(_textures, 2));
    %for i = 1:length(_textures)
    %    textures(i) = _textures(i);
    %end
    %
    %profile = cell(1, size(_profile, 1));
    %profile(1) = _profile(1, :);
    %profile(2) = _profile(2, :);

    profile = {[0, thickness, 0], [1, 2, 3]};
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);

    %res3(aa)
    %parm.res3.sens=1;
    %##parm.res3.gauss_x = 100

%    x = linspace(-period(1)/2, period(1)/2, 50);
%    y = linspace(-period(2)/2, period(2)/2, 50);
%    y = linspace(period(2)/2, -period(2)/2, 50);
    x = linspace(0, period(1), 11);
    y = linspace(period(2), 0, 11);

%    x = [0:1:49] * period(1) / 50 - period(1)/2;
%    x = [1:1:50] * period(1) / 50 - period(1)/2;
%    y = [0:1:49] .* period(2) / 50 - period(2)/2
%    y = [1:1:50] .* period(2) / 50 - period(2)/2
%    y = [50:-1:1] .* period(2) / 50 - period(2)/2
%    y = [49:-1:0] .* period(2) / 50 - period(2)/2

    parm.res3.trace=0;  %trace automatique % automatic trace

%    parm.res3.npts = res3_npts;

    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x,y,aa,profile,einc, parm);

%    if pol == 1  % TE
%        top_refl_info = res.TEinc_top_reflected;
%        top_tran_info = res.TEinc_top_transmitted;
%        bottom_refl_info = res.TEinc_bottom_reflected;
%        bottom_tran_info = res.TEinc_bottom_transmitted;
%    else  % TM
%        top_refl_info = res.TMinc_top_reflected;
%        top_tran_info = res.TMinc_top_transmitted;
%        bottom_refl_info = res.TMinc_bottom_reflected;
%        bottom_tran_info = res.TMinc_bottom_transmitted;
%    end

    top_refl_info_te = res.TEinc_top_reflected;
    top_tran_info_te = res.TEinc_top_transmitted;
    top_refl_info_tm = res.TMinc_top_reflected;
    top_tran_info_tm = res.TMinc_top_transmitted;

    bottom_refl_info_te = res.TEinc_bottom_reflected;
    bottom_tran_info_te = res.TEinc_bottom_transmitted;
    bottom_refl_info_tm = res.TMinc_bottom_reflected;
    bottom_tran_info_tm = res.TMinc_bottom_transmitted;
end

function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_5();

    warning('off', 'Octave:possible-matlab-short-circuit-operator');
    warning('off', 'Invalid UTF-8 byte sequences have been replaced.');
    warning('off', 'findstr is obsolete; use strfind instead');

    factor = 1;
    pol = 1;
    n_top = 1;
    n_bot = 1;
    theta = 0;
    phi = 0;
    nn = [11,11];
    period = [480/factor, 480/factor];
    wavelength = 550/factor;
    thickness = 220/factor;

    PatternIn = [0 0 0 0 0 0; 0 0 1 1 0 0; 0 1 0 0 1 0; 0 1 0 0 1 0; 0 0 1 1 0 0; 0 0 0 0 0 0];
%    PatternIn = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; 3, 3, 3, 3, 3, 1, 1, 1, 1, 1; ]
%    PatternIn = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1]
%    PatternIn = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;1, 1, 1, 1, 1, 0, 0, 0, 0, 0;]
%    PatternIn = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    XGrid = [3.5:1:8.5]/6 * period(1);
    YGrid = [3.5:1:8.5]/6 * period(1);

    XGrid = linspace(-period(1)/2 + period(1)/12, period(1)/2 - period(1)/12, 6) + period(1)/2;
    YGrid = linspace(-period(2)/2 + period(2)/12, period(2)/2 - period(2)/12, 6) + period(2)/2;
    YGrid = -linspace(-period(2)/2 + period(2)/12, period(2)/2 - period(2)/12, 6) - period(2)/2;

%    XGrid = linspace(-period(1)/2 + period(1)/12, period(1)/2 - period(1)/12, 6);
%    YGrid = -linspace(-period(2)/2 + period(2)/12, period(2)/2 - period(2)/12, 6);

%    XGrid = [0.5:1:9.5] * period(1) / 10
%    YGrid = [0.5:1:9.5] * period(2) / 10

    % RCWA

    retio;
    textures = cell(1,3);
    textures{1} = {n_top};
    textures{2} = FractureGeom(PatternIn,1,4,XGrid,YGrid);
    textures{3} = {n_bot};
    profile = {[0, thickness, 0], [1, 2, 3]};

    parm = res0;
    parm.res1.champ = 1;      % calculate precisely
    parm.res1.trace = 0;

    k_parallel = n_top*sind(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

    parm = res0;

    parm.not_io = 1;  % no write data on hard disk
    parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%    parm.res3.npts=[0,1,0];
    parm.res3.npts=[11,11,11];

    %parm.res1.trace = 1; % show the texture
    %
    %textures = cell(1, size(_textures, 2));
    %for i = 1:length(_textures)
    %    textures(i) = _textures(i);
    %end
    %
    %profile = cell(1, size(_profile, 1));
    %profile(1) = _profile(1, :);
    %profile(2) = _profile(2, :);

    profile = {[0, thickness, 0], [1, 2, 3]};
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);

    %res3(aa)
    %parm.res3.sens=1;
    %##parm.res3.gauss_x = 100

    x = linspace(-period(1)/2, period(1)/2, 11);
    y = linspace(-period(2)/2, period(2)/2, 11);
    y = linspace(period(2)/2, -period(2)/2, 11);


    x = linspace(0, period(1), 11);
    y = linspace(period(2), 0, 11);

%    x = [0:1:49] * period(1) / 50 - period(1)/2;
%    x = [1:1:50] * period(1) / 50 - period(1)/2;
%    y = [0:1:49] .* period(2) / 50 - period(2)/2
%    y = [1:1:50] .* period(2) / 50 - period(2)/2
%    y = [50:-1:1] .* period(2) / 50 - period(2)/2
%    y = [49:-1:0] .* period(2) / 50 - period(2)/2

    parm.res3.trace=0;  %trace automatique % automatic trace

%    parm.res3.npts = res3_npts;

    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x,y,aa,profile,einc, parm);

%    if pol == 1  % TE
%        top_refl_info = res.TEinc_top_reflected;
%        top_tran_info = res.TEinc_top_transmitted;
%        bottom_refl_info = res.TEinc_bottom_reflected;
%        bottom_tran_info = res.TEinc_bottom_transmitted;
%    else  % TM
%        top_refl_info = res.TMinc_top_reflected;
%        top_tran_info = res.TMinc_top_transmitted;
%        bottom_refl_info = res.TMinc_bottom_reflected;
%        bottom_tran_info = res.TMinc_bottom_transmitted;
%    end

    top_refl_info_te = res.TEinc_top_reflected;
    top_tran_info_te = res.TEinc_top_transmitted;
    top_refl_info_tm = res.TMinc_top_reflected;
    top_tran_info_tm = res.TMinc_top_transmitted;

    bottom_refl_info_te = res.TEinc_bottom_reflected;
    bottom_tran_info_te = res.TEinc_bottom_transmitted;
    bottom_refl_info_tm = res.TMinc_bottom_reflected;
    bottom_tran_info_tm = res.TMinc_bottom_transmitted;
end

function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_6();

    warning('off', 'Octave:possible-matlab-short-circuit-operator');
    warning('off', 'Invalid UTF-8 byte sequences have been replaced.');
    warning('off', 'findstr is obsolete; use strfind instead');

    factor = 1;
    pol = 1;
    n_top = 1;
    n_bot = 1;
    theta = 10;
    phi = 20;
    nn = [11,11];
    period = [480/factor, 480/factor];
    wavelength = 550/factor;
    thickness = 220/factor;


    a = [0+240, 120+240, 160, 80, 4, 1];
    b  = [0+240, -120+240, 160, 80, 4, 1];
    c = [120+240, 0+240, 80, 160, 4, 1];
    d = [-120+240, 0+240, 80, 160, 4, 1];

    textures = cell(1,3);
    textures{1} = n_top;
    textures{2} = {1,a, b, c, d};
    textures{3} = n_bot;


    parm = res0;
    parm.res1.champ = 1;      % calculate precisely
    parm.res1.trace = 0;

    k_parallel = n_top*sind(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

    parm = res0;

    parm.not_io = 1;  % no write data on hard disk
    parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%    parm.res3.npts=[0,1,0];
    parm.res3.npts=[11,11,11];

    %parm.res1.trace = 0; % show the texture
    %
    %textures = cell(1, size(_textures, 2));
    %for i = 1:length(_textures)
    %    textures(i) = _textures(i);
    %end
    %
    %profile = cell(1, size(_profile, 1));
    %profile(1) = _profile(1, :);
    %profile(2) = _profile(2, :);

    profile = {[0, thickness, 0], [1, 2, 3]};
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);

    %res3(aa)
    %parm.res3.sens=1;
    %##parm.res3.gauss_x = 100

%    x = linspace(-period(1)/2, period(1)/2, 50);
%    y = linspace(-period(2)/2, period(2)/2, 50);
%    y = linspace(period(2)/2, -period(2)/2, 50);
    x = linspace(0, period(1), 11);
    y = linspace(period(2), 0, 11);

%    x = [0:1:49] * period(1) / 50 - period(1)/2;
%    x = [1:1:50] * period(1) / 50 - period(1)/2;
%    y = [0:1:49] .* period(2) / 50 - period(2)/2
%    y = [1:1:50] .* period(2) / 50 - period(2)/2
%    y = [50:-1:1] .* period(2) / 50 - period(2)/2
%    y = [49:-1:0] .* period(2) / 50 - period(2)/2

    parm.res3.trace=0;  %trace automatique % automatic trace

%    parm.res3.npts = res3_npts;

    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x,y,aa,profile,einc, parm);

%    if pol == 1  % TE
%        top_refl_info = res.TEinc_top_reflected;
%        top_tran_info = res.TEinc_top_transmitted;
%        bottom_refl_info = res.TEinc_bottom_reflected;
%        bottom_tran_info = res.TEinc_bottom_transmitted;
%    else  % TM
%        top_refl_info = res.TMinc_top_reflected;
%        top_tran_info = res.TMinc_top_transmitted;
%        bottom_refl_info = res.TMinc_bottom_reflected;
%        bottom_tran_info = res.TMinc_bottom_transmitted;
%    end

    top_refl_info_te = res.TEinc_top_reflected;
    top_tran_info_te = res.TEinc_top_transmitted;
    top_refl_info_tm = res.TMinc_top_reflected;
    top_tran_info_tm = res.TMinc_top_transmitted;

    bottom_refl_info_te = res.TEinc_bottom_reflected;
    bottom_tran_info_te = res.TEinc_bottom_transmitted;
    bottom_refl_info_tm = res.TMinc_bottom_reflected;
    bottom_tran_info_tm = res.TMinc_bottom_transmitted;
end

function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reti_2d_7();

    warning('off', 'Octave:possible-matlab-short-circuit-operator');
    warning('off', 'Invalid UTF-8 byte sequences have been replaced.');
    warning('off', 'findstr is obsolete; use strfind instead');

    factor = 1;
    pol = 1;
    n_top = 1;
    n_bot = 1;
    theta = 10;
    phi = 20;
    nn = [2,2];
    period = [480/factor, 480/factor];
    wavelength = 550/factor;
    thickness = 220/factor;


    a = [0+240, 120+240, 160, 80, 4-1i, 1];
    b  = [0+240, -120+240, 160, 80, 4, 1];
    c = [120+240, 0+240, 80, 160, 4+5i, 1];
    d = [-120+240, 0+240, 80, 160, 4, 1];

    textures = cell(1,3);
    textures{1} = n_top;
    textures{2} = {1,a, b, c, d};
    textures{3} = n_bot;


    parm = res0;
    parm.res1.champ = 1;      % calculate precisely
    parm.res1.trace = 0;

    k_parallel = n_top*sind(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

    parm = res0;

    parm.not_io = 1;  % no write data on hard disk
    parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%    parm.res3.npts=[0,1,0];
    parm.res3.npts=[11,11,11];

    %parm.res1.trace = 0; % show the texture
    %
    %textures = cell(1, size(_textures, 2));
    %for i = 1:length(_textures)
    %    textures(i) = _textures(i);
    %end
    %
    %profile = cell(1, size(_profile, 1));
    %profile(1) = _profile(1, :);
    %profile(2) = _profile(2, :);

    profile = {[0, thickness, 0], [1, 2, 3]};
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);

    %res3(aa)
    %parm.res3.sens=1;
    %##parm.res3.gauss_x = 100

%    x = linspace(-period(1)/2, period(1)/2, 50);
%    y = linspace(-period(2)/2, period(2)/2, 50);
%    y = linspace(period(2)/2, -period(2)/2, 50);
    x = linspace(0, period(1), 11);
    y = linspace(period(2), 0, 11);

%    x = [0:1:49] * period(1) / 50 - period(1)/2;
%    x = [1:1:50] * period(1) / 50 - period(1)/2;
%    y = [0:1:49] .* period(2) / 50 - period(2)/2
%    y = [1:1:50] .* period(2) / 50 - period(2)/2
%    y = [50:-1:1] .* period(2) / 50 - period(2)/2
%    y = [49:-1:0] .* period(2) / 50 - period(2)/2

    parm.res3.trace=0;  %trace automatique % automatic trace

%    parm.res3.npts = res3_npts;

    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x,y,aa,profile,einc, parm);

%    if pol == 1  % TE
%        top_refl_info = res.TEinc_top_reflected;
%        top_tran_info = res.TEinc_top_transmitted;
%        bottom_refl_info = res.TEinc_bottom_reflected;
%        bottom_tran_info = res.TEinc_bottom_transmitted;
%    else  % TM
%        top_refl_info = res.TMinc_top_reflected;
%        top_tran_info = res.TMinc_top_transmitted;
%        bottom_refl_info = res.TMinc_bottom_reflected;
%        bottom_tran_info = res.TMinc_bottom_transmitted;
%    end

    top_refl_info_te = res.TEinc_top_reflected;
    top_tran_info_te = res.TEinc_top_transmitted;
    top_refl_info_tm = res.TMinc_top_reflected;
    top_tran_info_tm = res.TMinc_top_transmitted;

    bottom_refl_info_te = res.TEinc_bottom_reflected;
    bottom_tran_info_te = res.TEinc_bottom_transmitted;
    bottom_refl_info_tm = res.TMinc_bottom_reflected;
    bottom_tran_info_tm = res.TMinc_bottom_transmitted;
end

% Divides the given geometry into rectangles to be used in Reticolo
function GeometryOut = FractureGeom(PatternIn,nLow,nHigh,XGrid,YGrid)

    % Acceptable refractive index tolerance in fracturing

    % Extract grid parameters
    dX = abs(XGrid(2)-XGrid(1));
    dY = abs(YGrid(2)-YGrid(1));
    [Nx, Ny] = size(PatternIn);

    Geometry = {nLow}; %Define background index

    % Fracture non binarized pixels
    for i = 1:Nx % Defining texture for patterned layer. Probably could have vectorized this.
        for j = 1:Ny
            if PatternIn(i,j) == 1
                 Geometry = [Geometry,{[XGrid(i),YGrid(j),dX,dY,nHigh,1]}];
            end
        end
    end
    GeometryOut = Geometry;
end