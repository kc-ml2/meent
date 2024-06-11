%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  [Input] None
%
%  [Output] Time <string> (ex. '2020.10.30_15.13.05')
%
%  [Description] 
%  Returns time string for filename.
%
%  [Ver. update] 2020-10-30
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function now_string = nowstring()

    date = clock;
    now_string = strcat(num2str(date(1)),'.',...
                      num2str(date(2)),'.',...
                      num2str(date(3)),'_',datestr(now,'HH.MM.SS'));
                  
end