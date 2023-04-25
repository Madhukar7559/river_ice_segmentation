function save_img(path, fname, ext, is_paper)
	out_path = fullfile(path, sprintf('%s.%s', fname, ext));
	if ~is_paper
		export_fig(out_path, '-transparent', ' -painters');
	else
		% export_fig(out_path);
		exportgraphics(gca, out_path)
	end
	out_fig_path = fullfile(path, sprintf('%s.fig', fname));
    savefig(out_fig_path)
end
