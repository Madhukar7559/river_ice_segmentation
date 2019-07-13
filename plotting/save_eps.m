function save_eps(path, fname, ext)
if nargin < 3
    ext = 'eps';
end
    export_fig(fullfile(path, sprintf('%s.%s', fname, ext)), '-transparent')
    savefig(fullfile(path, sprintf('%s.fig', fname)))
end
