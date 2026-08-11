function restore_workdir = reticolo_setup(script_dir)
%RETICOLO_SETUP Configure RETICOLO and enter a coefficient case directory.
%
% RETICOLO_ROOT may point at any RETICOLO installation. If it is unset, walk up from the
% case directory looking for a sibling <parent>/reticolo/reticolo checkout, so the same
% script works whatever depth the case folder sits at. The returned onCleanup object
% restores MATLAB's original directory when the caller script finishes.

reticolo_root = getenv('RETICOLO_ROOT');

if isempty(reticolo_root)
    % Search upward rather than counting '..' levels: cases live at different depths
    % (lossless/<case>/ and lossy/<case>/), and a hard-coded count silently resolves to
    % the wrong directory when that changes.
    d = script_dir;
    while true
        candidate = fullfile(d, '..', 'reticolo', 'reticolo');
        if isfile(fullfile(candidate, 'res1.m'))
            reticolo_root = candidate;
            break
        end
        parent = fileparts(d);
        if strcmp(parent, d)        % reached the filesystem root
            break
        end
        d = parent;
    end
end

if isempty(reticolo_root) || ~isfile(fullfile(reticolo_root, 'res1.m'))
    error(['RETICOLO not found. Set RETICOLO_ROOT to the directory containing ' ...
           'res1.m, or place the reticolo checkout beside the meent one.']);
end
addpath(genpath(reticolo_root));

% Always place generated references beside the case notebook, regardless of MATLAB's
% starting directory, and restore the caller's directory when the case script finishes.
previous_workdir = pwd;
restore_workdir = onCleanup(@() cd(previous_workdir));
cd(script_dir);
end
