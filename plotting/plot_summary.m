clear all;

colRGBDefs;

% plot_title='UNet';
% k=importdata('unet_summary.txt');

% plot_title='SegNet';
% k=importdata('segnet_summary.txt');

% plot_title='Deeplab';
% k=importdata('deeplab_summary.txt');

% plot_title='DenseNet';
% k=importdata('densenet_summary.txt');

% plot_title='Comparing models';
plot_title='Selective Training';

% y_label = 'Recall (%)';
% y_label = 'pixel accuracy';
y_label = 'acc/IOU';

% plot_title='Recall rates using 5000 video images for training';
% plot_title='Recall rates on 20K 3-class test set without static images';
fname = 'combined_summary.txt';
line_width = 3;
transparent_bkg = 1;
transparent_legend = 0;
vertcal_x_label = 0;
axes_font_size = 18;
legend_font_size = 20;
title_font_size = 30;
bar_plot = 0;

rec_prec_mode = 0;
enable_ap = 0;
thresh_mode = 2;

% markers = {'o', '+', '*', 'x', 'p', 'd', 'o', '+'};
% markers = {'o', 'o', '+', '+', '*', '*', 'x', 'x', 'p', 'p', 'd', 'd'};

% markers = {'.', '.', '.', '.', '.', '.', '.', '.', '.', '.'};

% line_specs = {'-or', '-+g', '--*r', '-+g', '--xg'};

% line_cols = {'red', 'blue', 'forest_green', 'magenta', 'red', 'blue', 'forest_green', 'magenta'};
% line_cols = {'red', 'blue', 'forest_green', 'magenta', 'cyan', 'peach_puff', 'green', 'black', 'maroon'};

% line_cols = {'forest_green', 'red', 'blue', 'magenta', 'cyan', 'green', 'peach_puff', 'black', 'maroon'};
% line_cols = {'forest_green', 'forest_green', 'red', 'red', 'blue', 'blue', 'magenta', 'magenta', 'cyan', 'cyan', 'green', 'green'};

% line_cols = {'red', 'magenta', 'blue', 'cyan', 'forest_green', 'green', 'peach_puff', 'black', 'maroon'};
% line_cols = {'red','red', 'magenta', 'magenta', 'blue', 'blue', 'cyan', 'cyan', 'forest_green', 'forest_green', 'green', 'peach_puff', 'black', 'maroon'};
line_cols = {'red','red', 'blue', 'blue', 'forest_green', 'magenta', 'magenta', 'cyan', 'cyan', 'forest_green', 'forest_green', 'green', 'peach_puff', 'black', 'maroon'};

% line_cols = {'red','red','red',...
%     'magenta', 'magenta',...
%     'blue', 'blue', 'blue',...
%     'cyan', 'cyan',...
%     'forest_green', 'forest_green', 'forest_green',...
%     'green', 'peach_puff', 'black', 'maroon'};

% line_cols = {'forest_green', 'blue', 'red', 'magenta', 'cyan'};
% line_cols = {'forest_green','forest_green', 'blue', 'blue', 'red', 'red', 'purple', 'purple', 'cyan', 'cyan'};

% line_cols = {'blue', 'forest_green', 'magenta', 'cyan'};
% line_cols = {'red', 'forest_green', 'blue', 'blue'};
% line_cols = {'forest_green', 'red'};

line_styles = {'-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'};
% line_styles = {'-', '-', '-', '-', '--', '--', '--', '--'};

% line_styles = {'--', '-', '-', '-', '-'};
% line_styles = {'-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':'};
line_styles = {'-', '--', '-', '--', ':', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--'};

% line_styles = {'-', '-', '--', '--', '--'};
% line_styles = {'-', '-', '--', '-'};

% line_styles = {'-o', '-+', '-*', '-x', '-s', '-p'};


% line_specs = {'-og', '-+r'};
% line_specs = {'--or', '-+g', '--xm', '-xm'};

% line_specs = {'-or', '-+g', '-*b', '--xm'};

% line_specs = {'-or', ':+r', '-*b', ':xb'};
% line_specs = {'-or', '--+r', '-*b', '--xb'};

% line_specs = {'-or', '-+g', '--*b', '--xm'};
% line_specs = {'--or', '-+b', '-*g', '-xm'};

% line_specs = {'--or', '-+g', '-*b', '-xm', '-sc', '-pk'};

% line_specs = {'-or', '-+g', '-*b',...
%     '--or', '--+g', '--*b'};

% line_specs = {'-+g', '-*b', '-xm',...
%     '--+g', '--*b', '--xm'};

% line_specs = {'-or', '-+g', '-*b', '-xm',...
%     '--or', '--+g', '--*b', '--xm'};

% line_specs = {'-or', '-+g', '-*b', '-xm', '-sc',...
%     '--or', '--+g', '--*b', '--xm', '--sc'};

% line_specs = {'-or', '--*r', '-+g', '--xg'};
% line_specs = {'-or', '-+g', '--*r', '-+g', '--xg'};

% set(0,'DefaultAxesFontName', 'Times New Roman');
% k=importdata('radon.txt');

set(0,'DefaultAxesFontSize', axes_font_size);
set(0,'DefaultAxesFontWeight', 'bold');

k = importdata(fname);

if bar_plot
    figure;
    bar(k.data);
else
    if rec_prec_mode
        n_items = size(k.data, 1)
        
        if thresh_mode
            n_lines = int32(size(k.data, 2) / 3)
        else
            n_lines = int32(size(k.data, 2) / 2)
        end
        
        x_data = zeros(n_items, n_lines);
        y_data = zeros(n_items, n_lines);
        
        fileID = fopen(fname,'r');
        plot_title = fscanf(fileID,'%s', 1);
        plot_legend = cell(n_lines, 1);
        for line_id = 1:n_lines
            plot_legend{line_id} = fscanf(fileID,'%s', 1);
            
            if thresh_mode
                thresh_data = k.data(:, 3*(line_id-1)+1);
                rec_data = k.data(:, 3*(line_id-1)+2);
                prec_data = k.data(:, 3*(line_id-1)+3);
                if thresh_mode==1
                    x_data(:, line_id) = thresh_data;
                    y_data(:, line_id) = rec_data;
                elseif thresh_mode==2
                    x_data(:, line_id) = thresh_data;
                    y_data(:, line_id) = prec_data;
                elseif thresh_mode==3
                    x_data(:, line_id) = rec_data;
                    y_data(:, line_id) = prec_data;
                else
                    error('Invalid thresh_mode: %d', thresh_mode)
                end
            else
                rec_data = k.data(:, 2*line_id-1);
                prec_data = k.data(:, 2*line_id);
                x_data(:, line_id) = rec_data;
                y_data(:, line_id) = prec_data;
            end
        end
        if enable_ap
            [ap, mrec, mprec] = VOCap(flipud(rec_data/100.0),...
                flipud(prec_data/100.0));
            x_data = mrec*100.0;
            y_data = mprec*100.0;
            ap = ap*100;
            fprintf('%s ap: %f%%\n', plot_legend{line_id}, ap);
        end
        % plot_legend
        fclose(fileID);

        if thresh_mode
            thresh_label = k.colheaders(1);
            rec_label = k.colheaders(2);
            prec_label = k.colheaders(3);                
            if thresh_mode==1
                x_label = thresh_label;
                y_label = rec_label;
            elseif thresh_mode==2
                x_label = thresh_label;
                y_label = prec_label;
            elseif thresh_mode==3
                x_label = rec_label;
                y_label = prec_label;
            else
                error('Invalid thresh_mode: %d', thresh_mode)
            end
        else
            x_label = k.colheaders(1);
            y_label = k.colheaders(2);
        end
%         x_ticks = zeros(n_lines, 1);
%         xtick_labels = cell(n_lines, 1);
%         for item_id = 1:n_items
%             xtick_labels{item_id} = sprintf('%d', 10*item_id);
%             x_ticks(item_id) = 10*item_id;
%         end
        %         x_ticks
        %         xtick_labels
    else
        if isfield(k,'colheaders')
            n_lines = size(k.data, 2) - 1;
            y_data = k.data(:, 2:end);
            % x_ticks = k.data(:, 1);
            x_data = repmat(k.data(:, 1),1, n_lines);
            plot_legend = {k.colheaders{2:end}};
            plot_title = k.textdata{1, 1};
            y_label = sprintf('%s', plot_legend{1});
            for line_id = 2:n_lines
                y_label = sprintf('%s/%s', y_label, plot_legend{line_id})
            end
            x_label = k.textdata{2, 1};
        else
            n_lines = size(k.data, 2)
            n_items = size(k.data, 1)
            y_data = k.data;
            %     x_label='Model';
            
            n_text_lines = size(k.textdata, 2)
            n_text_items = size(k.textdata, 1)
            if n_text_items == n_items + 3
                y_label = k.textdata(1, 1)
                k.textdata = k.textdata(2:end, :);
                n_text_items = n_text_items - 1;
            end
            if n_text_items == n_items + 2
                plot_title = k.textdata(1, 1)
                k.textdata = k.textdata(2:end, :);
            end
            x_label = k.textdata(1, 1)
            plot_legend = {k.textdata{1, 2:end}}
            xtick_labels = k.textdata(2:end, 1)
            x_ticks = 1:n_items;
            x_data = repmat((1:n_items)',1, n_lines);
            
            for j = 1:n_items
                if xtick_labels{j}(1)=='_'
                    xtick_labels{j} = xtick_labels{j}(2:end);
                end
            end
        end
    end
    figure
    %     y_data
    %     x_data
    %     line_cols
    %     line_styles
    % n_lines
    final_legend = {};
    for i = 1:n_lines
        y_datum = y_data(:, i);
        x_datum = x_data(:, i);
        line_col = line_cols{i};
        line_style = line_styles{i};
        if exist('markers', 'var')            
            marker = markers{i};
        else
            marker = 'none';
        end
        vis = 'on';
        line_width_ = line_width;
        
        if strcmp(plot_legend{i}, '_')
            fprintf('Turning off legend for line %d\n', i)
            vis = 'off';
            line_width_ = 2;
            marker = 'none';
            line_style = ':';
        elseif strcmp(plot_legend{i}, '__')
            fprintf('Turning off legend for line %d\n', i)
            vis = 'off';
            marker = 'none';
            line_style = '--';
        else
            final_legend{end+1} = plot_legend{i};
        end
        %     line_spec = line_specs{i};
        plot(x_datum, y_datum,...
            'Color', col_rgb{strcmp(col_names,line_col)},...
            'LineStyle', line_style,...
            'LineWidth', line_width_,...
            'Marker', marker,...
            'HandleVisibility', vis);
        hold on
    end
    hold off
    
    h_legend=legend(final_legend, 'Interpreter','none');
    set(h_legend,'FontSize',legend_font_size);
    set(h_legend,'FontWeight','bold');
    grid on;
    
    % ax = gca;
    % ax.GridAlpha=0.25;
    % ax.GridLineStyle=':';
    % set (gca, 'GridAlphaMode', 'manual');
    % set (gca, 'GridAlpha', 0.5);
    % set (gca, 'GridLineStyle', '-');
    
    try
        if exist('xtick_labels', 'var')
            xticks(x_ticks);
        end
        if exist('xtick_labels', 'var')
            xticklabels(xtick_labels);
        end
    catch
        if exist('xtick_labels', 'var')
            set(gca, 'XTick', x_ticks)
        end
        
        if exist('xtick_labels', 'var')
            set(gca, 'xticklabel', xtick_labels)
        end
    end
    % ylabel('metric value');
    y_label = strtrim(y_label);
    ylabel(y_label, 'fontsize',20, 'FontWeight','bold', 'Interpreter', 'none');
    
    x_label = strtrim(x_label);
    xlabel(x_label, 'fontsize',20, 'FontWeight','bold', 'Interpreter', 'none');
    if vertcal_x_label
        xticklabel_rotate([],90,[], 'fontsize',20, 'FontWeight','bold', 'Interpreter', 'none');
    end
    % ylim([0.60, 0.90]);
    % ylim([0.65, 0.90]);
    plot_title = strtrim(plot_title);
    title(plot_title, 'fontsize',title_font_size, 'FontWeight','bold', 'Interpreter', 'none');
    if transparent_bkg
        set(gca,'color','none')
        if transparent_legend
            set(h_legend,'color','none');
        end
    end
end




