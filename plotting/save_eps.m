function save_eps(path, fname)
    export_fig(fullfile(path, sprintf('%s.eps', fname)), '-transparent')
    savefig(fullfile(path, sprintf('%s.fig', fname)))
end
