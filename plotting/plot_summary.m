if ~exist('no_clear', 'var')
    clear all;
end

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
% y_label = 'acc/IOU';
y_label = 'Recall / Precision';

% plot_title='Recall rates using 5000 video images for training';
% plot_title='Recall rates on 20K 3-class test set without static images';
fname = 'combined_summary.txt';
line_width = 3;
transparent_bkg = 1;
transparent_legend = 1;
vertcal_x_label = 0;
colored_x_label = 0;
axes_font_size = 18;
legend_font_size = 30;
title_font_size = 30;
bar_plot = 0;
title_interpreter = 'tex';
% title_interpreter = 'none';
rec_prec_mode = 0;
enable_ap = 0;
thresh_mode = 0;
white_text = 1;
y_limits = [0, 100];
add_ylabel_to_title = 0;

enable_y_label = 1;
enable_x_label = 0;
mode = 0;
if mode == 0
    line_cols = {'blue', 'forest_green', 'red', 'purple', 'magenta', 'cyan',...
        'green', 'maroon', 'peach_puff', 'black'};
    line_styles = {'-', '-', '-', '-', '-', '-', '-', '-', '-'};
    markers = {'o', '+', '*', 'x', 'p', 'd', 'o','+', '*', 'x'};

elseif mode == 1
%     line_cols = {'blue', 'forest_green', 'blue', 'forest_green'};
    line_cols = {'blue', 'purple', 'blue', 'purple'};
    line_styles = {'-', '-', '-', ':', ':', ':'};
    markers = {'o', '+','o', '+'};
    % valid_columns = [1, 3];
elseif mode == 2
    line_cols = {'blue', 'purple', 'blue', 'purple'};
    line_styles = {'-', '-', ':', ':', ':'};
    markers = {'o', '+', 'o', '+'};
elseif mode == 3
%         line_cols = {'blue', 'blue', 'forest_green', 'forest_green',...
%             'magenta', 'magenta', 'red', 'red',...
%             'cyan', 'cyan', 'purple', 'purple'};
        
%         line_cols = {'blue', 'red', 'forest_green', 'cyan','purple','green',...
%             'blue', 'forest_green', 'green','red', 'cyan','purple'};
        
        line_cols = {'blue', 'blue', 'blue',...
            'red','red','red',...
            'forest_green',...
            'forest_green', 'forest_green',...
            'purple', 'purple','purple'};
        
        
        line_styles = {
            '-', '-', '-', '-',...
            '-', '-', '-', '-',...
            '-', '-','-', '-'...
            '-', '-', '-', '-'};
        markers = {
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            };
        
elseif mode == 4
%         line_cols = {'blue', 'blue', 'forest_green', 'forest_green',...
%             'magenta', 'magenta', 'red', 'red',...
%             'cyan', 'cyan', 'purple', 'purple'};
        
%         line_cols = {'blue', 'red', 'forest_green', 'cyan','purple','green',...
%             'blue', 'forest_green', 'green','red', 'cyan','purple'};
        
        line_cols = {'blue', 'blue', 'blue', 'blue',...
            'red','red','red','red',...
            'forest_green',...
            'forest_green', 'forest_green', 'forest_green',...
            'purple', 'purple','purple','purple'};
        
        
        line_styles = {
            '-', '-', '-', '-',...
            '-', '-', '-', '-',...
            '-', '-','-', '-'...
            '-', '-', '-', '-'};
        markers = {
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            };
end
% line_styles = {'-', '-', '-', '--', '--', '--'};

% line_styles = {'--', '-', '-', '-', '-'};
% line_styles = {'-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':'};
% line_styles = {'-', '--', '-', '--', ':', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--'};

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

if white_text
    set(0,'DefaultAxesXColor', col_rgb{strcmp(col_names,'white')});
    set(0,'DefaultAxesYColor', col_rgb{strcmp(col_names,'white')});
end

if white_text
    set(0,'DefaultAxesLineWidth', 2.0);
    set(0,'DefaultAxesGridLineStyle', ':');    
%     set(0,'DefaultAxesGridAlpha', 1.0);
%     set(0, 'DefaultAxesGridColor', col_rgb{strcmp(col_names,'white')});
%     set(0,'DefaultAxesMinorGridColor', col_rgb{strcmp(col_names,'white')});
end

 

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
            n_items = size(k.data, 1);
            n_lines = size(k.data, 2) - 1;
            y_data = k.data(:, 2:end);
            % x_ticks = k.data(:, 1);
            x_data = repmat(k.data(:, 1),1, n_lines);
            plot_legend = {k.colheaders{2:end}};
            plot_title = k.textdata{1, 1};
            y_label = sprintf('%s', plot_legend{1});
            for line_id = 2:n_lines
                if ~strcmp(plot_legend{line_id}, '_') ||  ~strcmp(plot_legend{line_id}, '__')
                    y_label = sprintf('%s/%s', y_label, plot_legend{line_id});
                end
            end
            x_label = k.textdata{2, 1};
        elseif size(k.textdata, 2) == size(k.data, 2) + 1
            n_items = size(k.data, 1);
            n_lines = size(k.data, 2);
            y_data = k.data(:, 1:end);
            % x_ticks = k.data(:, 1);
            x_data = repmat(transpose(1:n_items),1, n_lines);
            plot_legend = {k.textdata{3, 2:end}};
            plot_title = k.textdata{1, 1};
            x_label = k.textdata{2, 1};
            xtick_labels = k.textdata(4:end, 1);
            x_ticks = 1:n_items;
            xlim([1, n_items]);
            
            if exist('valid_columns', 'var')
                x_data = x_data(:, valid_columns);
                y_data = y_data(:, valid_columns);
                plot_legend = plot_legend(:, valid_columns);
                n_lines = size(x_data, 2);
                
                line_cols = line_cols(valid_columns);
                line_styles = line_styles(valid_columns);
                markers = markers(valid_columns);
                temp = strsplit(x_label, '---');
                
                x_label = temp{1};
                y_label = temp{valid_columns(1)+1};     
                
                if add_ylabel_to_title
                    plot_title = sprintf('%s %s', plot_title, y_label);     
                end
            else
                y_label = sprintf('%s', plot_legend{1});
                for line_id = 2:n_lines
                    if ~strcmp(plot_legend{line_id}, '_') ||  ~strcmp(plot_legend{line_id}, '__')
                        y_label = sprintf('%s/%s', y_label, plot_legend{line_id});
                    end
                end
            end
            
        else
            n_lines = size(k.data, 2);
            n_items = size(k.data, 1);
            y_data = k.data;
            %     x_label='Model';
            
            n_text_lines = size(k.textdata, 2);
            n_text_items = size(k.textdata, 1);
            if n_text_items == n_items + 3
                y_label = k.textdata(1, 1)
                k.textdata = k.textdata(2:end, :);
                n_text_items = n_text_items - 1;
            end
            if n_text_items == n_items + 2
                plot_title = k.textdata(1, 1);
                k.textdata = k.textdata(2:end, :);
            end
            x_label = k.textdata(1, 1);
            plot_legend = {k.textdata{1, 2:end}};
            xtick_labels = k.textdata(2:end, 1);
            x_ticks = 1:n_items;
            x_data = repmat((1:n_items)',1, n_lines);           

        end
    end
    
    for j = 1:n_items
        if xtick_labels{j}(1)=='_'
            xtick_labels{j} = xtick_labels{j}(2:end);
        end
    end
    figure_handle = figure;
    propertyeditor(figure_handle,'on');
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
%             vis = 'off';
            line_width_ = 2;
            marker = 'none';
            line_style = ':';
        elseif strcmp(plot_legend{i}, '__')
            fprintf('Turning off legend for line %d\n', i)
%             vis = 'off';
            marker = 'none';
            line_style = '--';
            line_width_=3;
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
    if white_text
        set(h_legend,'TextColor', col_rgb{strcmp(col_names,'white')});
        set(h_legend,'LineWidth', 1.0);
    end
%     grid on;   
    
    grid(gca,'on')

    % ax = gca;
    % ax.GridAlpha=0.25;
    % ax.GridLineStyle=':';
%     set (gca, 'GridAlphaMode', 'manual');
%     set (gca, 'GridAlpha', 1.0);
    % set (gca, 'GridLineStyle', '-');
    
    xtick_label_cols = cell(length(xtick_labels), 1);
    for x_label_id=1:length(xtick_labels)
        curr_x_label=xtick_labels{x_label_id};
        
        split_x_label = strsplit(curr_x_label, '---');       
        
        
        if length(split_x_label)==2
            new_x_label = split_x_label{1};
            xtick_label_col = split_x_label{2};            
        else
            new_x_label = curr_x_label;
            xtick_label_col='black';
        end
        
%         
%         if contains(curr_x_label, '@')
%             split_x_label = split(curr_x_label, '@');
%             new_x_label = split_x_label{1};
%             xtick_label_col = split_x_label{2};            
%         else
%             new_x_label = curr_x_label;
%             xtick_label_col='black';
%         end
%         
        xtick_labels{x_label_id} = new_x_label;
        xtick_label_cols{x_label_id} = col_rgb{strcmp(col_names,xtick_label_col)};        
    end
    
    xlim([1, n_items]);
    
    ax = gca;
    ax.LineWidth = 4;
    ax.GridAlpha = 4;

    try
        if exist('x_ticks', 'var')
            xticks(x_ticks);
        end
        if exist('xtick_labels', 'var')
            xticklabels(xtick_labels);
        end
    catch
        if exist('x_ticks', 'var')
            set(gca, 'XTick', x_ticks)
        end
        
        if exist('xtick_labels', 'var')
            set(gca, 'xticklabel', xtick_labels)
        end
    end
    % ylabel('metric value');
    y_label = strtrim(y_label);
    if enable_y_label
        ylabel(y_label, 'fontsize',20, 'FontWeight','bold', 'Interpreter', 'none');
    end
    
%     ax = gca;
%     outerpos = ax.OuterPosition;
%     ti = ax.TightInset; 
%     left = outerpos(1) + ti(1);
%     bottom = outerpos(2) + ti(2);
%     ax_width = outerpos(3) - ti(1) - ti(3);
%     ax_height = outerpos(4) - ti(2) - ti(4);
%     ax.Position = [left bottom ax_width ax_height];
%     ax.Position = outerpos;    

    x_label = strtrim(x_label);
    if enable_x_label
        xlabel(x_label, 'fontsize',20, 'FontWeight','bold', 'Interpreter', 'none');
    end
    if colored_x_label
        xticklabel_rotate([],0,[], xtick_label_cols, 'fontsize',20, 'FontWeight','bold',...
            'Interpreter', 'none');
    elseif vertcal_x_label
        xticklabel_rotate([],90,[], xtick_label_cols, 'fontsize',20, 'FontWeight','bold',...
            'Interpreter', 'none');
    end
    % ylim([0.60, 0.90]);
    
    if exist('y_limits', 'var')
        ylim(y_limits);
    end
    
    plot_title = strtrim(plot_title);
    title_obj = title(plot_title, 'fontsize',title_font_size, 'FontWeight','bold',...
        'Interpreter', title_interpreter);    
    if white_text
        set(title_obj,'Color', col_rgb{strcmp(col_names,'white')});
    end
    if transparent_bkg
        set(gca,'color','none')
        if transparent_legend
            set(h_legend,'color','none');
        end
    end
    
%     if white_text
%         ax = gca;
%         set(0,'DefaultAxesGridAlpha', 1.0);
%         set(0, 'DefaultAxesGridColor', col_rgb{strcmp(col_names,'white')});
%         set(0,'DefaultAxesMinorGridColor', col_rgb{strcmp(col_names,'white')});
%     end
%     
end




