classdef UtilsPlotting
    % class to collect helper functions to visualize results

    properties
        my_fontsize = 40;
    end

    methods 
        function [fig, ax] = plot_lineplot(obj, x_data, y_data, my_title, my_xlabel, my_ylabel, varargin)
            % generate standard line plot
            % x_data: [1 , N], data to plot along x-axis
            % y_data: [M , N], data to plot along y-axis
            % my_title, my_xlabel, my_ylabel: (str), axis labels for title, x- and y-axis
            
            my_mad = [];
            display_names = {};               
            optargin = size(varargin,2);
            for i = 1:2:optargin
                switch varargin{i}
                    case 'display_names'
                        display_names = varargin{i+1};
                    case 'my_mad'
                        my_mad = varargin{i+1};
                    otherwise
                        assert (false, 'Option not recognized.');
                end
            end
            
            % format y_data
            if size(y_data,1)>size(y_data,2)
                y_data = y_data.';
            end            
            
            % plot
            fig = figure('Position', [10 10 1400 1400]);
            ax = subplot(1,1,1); grid on; hold on;
            for data_nr = 1:(numel(y_data)/numel(x_data))
                if ~isempty(display_names)
                    display_name = display_names{data_nr};
                else
                    display_name = '';
                end
                plot(ax, x_data, y_data(data_nr, :), 'LineWidth', 3, ...
                    'DisplayName', display_name); hold on;
            end
            
            % add shaded area for MAD
            if ~isempty(my_mad)
                inBetween = [y_data+my_mad, fliplr(y_data-my_mad)];
                f = fill([x_data, fliplr(x_data)], inBetween,  'k', 'LineStyle', ...
                        'none', 'HandleVisibility', 'off'); hold on;
                f.FaceColor = 'k';
                f.FaceAlpha = 0.2;
            end
            
            % labels
            ax.FontSize = obj.my_fontsize - 10;
            title(ax, my_title, 'FontSize', obj.my_fontsize);
            xlabel(ax, my_xlabel, 'FontSize', obj.my_fontsize);
            ylabel(ax, my_ylabel, 'FontSize', obj.my_fontsize); hold on;
            if ~isempty(display_names)
                legend('FontSize', obj.my_fontsize, 'Location', 'best'); hold on;
            end
            ylim(ax, [0, 1.1*max([my_mad(:); y_data(:)])]); hold on;
            
        end
        
        
        function [proj_sampling_locs] = project_sampling_locs_onto_PCs(obj, sampling_locs, PCs)
            % get projection of n-dimensional sampling location vectors
            % onto 2/3-dimensional principal component dimensions
            % sampling_loc: [n_input_conditions, n_units, n_start_pts_per_inpCond], sampling locations for opdims
            % PCs: [n_units, n_pcs], principal components of network activity
            
            n_inpConds = size(sampling_locs,1);
            n_sampling_locs_per_conds = size(sampling_locs,3);
            n_PCs_to_proj = size(PCs, 2);

            proj_sampling_locs = nan([n_inpConds, n_sampling_locs_per_conds, n_PCs_to_proj]);
            for inpCond_nr = 1:n_inpConds
                for start_pt_nr = 1:n_sampling_locs_per_conds
                    sampl_loc = sampling_locs(inpCond_nr, :, start_pt_nr);
                    for PC_nr = 1:n_PCs_to_proj
                        proj_sampling_locs(inpCond_nr, start_pt_nr, PC_nr) = dot(PCs(:,PC_nr), sampl_loc(:));
                    end
                end
            end
            
        end

        
        function [PCs] = get_PCs_of_network_activity(obj, network_activity)
            % get principal components of network_activity
            % network_activity: [n_units, n_timesteps, n_trials]
            
            [n_units, n_timesteps, n_trials] = size(network_activity);

            all_netActivity = nan([n_units, n_trials*n_timesteps]);        
            for trial_nr = 1:n_trials
                all_netActivity(:, (trial_nr-1)*n_timesteps+1:trial_nr*n_timesteps) = network_activity(:, :, trial_nr);
            end
            [PCs, ~, ~] = pca(all_netActivity.');
        end

        
        function [proj_condAvgTrajs] = project_condition_average_trajectories_onto_PCs(obj, ...
                                        network_activity, PCs, network_type, sampling_loc_props, ...
                                        all_freq_ids, conditionIds, coherencies_trial)
            % get projection of network activty onto PCs of network
            % activity, but sort by different input conditions
            
            % network_activity: [n_units, n_timesteps, n_trials]
            % PCs: [n_units, n_pcs], principal components of network activity
            % network_type: (str), 'swg' or 'ctxt' 
            % sampling_loc_props: (str) with properties of all sampling locations
            % all_freq_ids: [n_trials, 1], all frequency IDs of each trial
            % conditionIds = [1, n_trials], context ID per trial
            % coherencies_trial = [nIntegrators, n_trials], input coherencies of sensory input 1 and 2 over trials      

            % constants
            n_timesteps = size(network_activity, 2);
            n_PCs_to_proj = size(PCs, 2);
            if strcmp(network_type, 'swg')
                n_inpConds = numel(sampling_loc_props.freq_idx_per_inpCond);
            elseif strcmp(network_type, 'ctxt')
                n_inpConds = numel(sampling_loc_props.ctxt_per_inpCond);
            else
                assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")
            end
            
            % sort trials per input condition and proj onto PCs
            proj_condAvgTrajs = nan([n_inpConds, n_timesteps, n_PCs_to_proj]);  
            for inpCond_nr = 1:n_inpConds
                % sort trials per input condition
                if strcmp(network_type, 'swg')
                    valid_trial_ids = (all_freq_ids(:, 1)==sampling_loc_props.freq_idx_per_inpCond(1, inpCond_nr));
                    valid_trial_ids = find(valid_trial_ids);
                    
                elseif strcmp(network_type, 'ctxt')
                    valid_trial_ids = conditionIds==sampling_loc_props.ctxt_per_inpCond(inpCond_nr) & ...
                            (sign(coherencies_trial(1, :))==sampling_loc_props.signCoh1_per_inpCond(inpCond_nr)) & ...
                            (sign(coherencies_trial(2, :))==sampling_loc_props.signCoh2_per_inpCond(inpCond_nr));
                else
                    assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")
                end
                mean_traj = mean(network_activity(:, :, valid_trial_ids), 3);
                
                % proj onto PCs
                for PC_nr = 1:n_PCs_to_proj
                    for t_nr = 1:n_timesteps
                        proj_condAvgTrajs(inpCond_nr, t_nr, PC_nr) = dot(PCs(:, PC_nr), mean_traj(:, t_nr));
                    end
                end
            end

        end
        
        function plot_condAvgTrajs(obj, ax, proj_condAvgTrajs, color_per_inpCond, network_type, legend_on, line_style)
            % plot projection of network activities over time
            % ax: (matlab subplot object)
            % proj_condAvgTrajs: [n_input_conditions, n_timesteps, n_PCS], projected network activities to plot
            % color_per_inpCond: [n_input_conditions, 3], rgb values for each input condition
            % network_type: (str), 'swg' or 'ctxt' 
            % legend_on: (bool), if true: add legend to plot
            % line_style: (matlab line style), e.g. ':', '-', ...

            % constants
            n_inpConds = size(proj_condAvgTrajs,1);
            
            handle_vis = 'off';
            if legend_on
                handle_vis = 'on';
            end
            
            for inpCond_nr = 1:n_inpConds
                
                projs = squeeze(proj_condAvgTrajs(inpCond_nr, :, :));
                if strcmp(network_type, 'swg')
                    plot3(ax, projs(:, 1), projs(:, 2), projs(:,3), 'Color', color_per_inpCond(inpCond_nr, :), ...
                        'LineWidth', 3, 'DisplayName', strcat(num2str(inpCond_nr)), ...
                        'HandleVisibility', handle_vis, 'LineStyle', line_style); hold on;
                    lgd = legend; title(lgd, "input conditions"); hold on;
                
                elseif strcmp(network_type, 'ctxt')
                    line(ax, projs(:, 1), projs(:, 2), 'Color', color_per_inpCond(inpCond_nr, :), ...
                        'LineWidth', 3, 'DisplayName', strcat(num2str(inpCond_nr)), ...
                        'HandleVisibility', handle_vis, 'LineStyle', line_style); hold on;
                    % place legend outside without decreasing plot size
                    lgd=legend(ax, 'Location', 'eastoutside');
                    axP = get(gca,'Position'); set(gca, 'Position', axP)
                    title(lgd, "input conditions"); hold on;
                end
                
            end
        end
        
        
        function plot_sampling_locs(obj, ax, proj_sampling_locs, network_type)
            % plot projection of sampling locations
            % ax: (matlab subplot object)
            % proj_sampling_locs: [n_input_conditions, n_sampling_locs_per_conds, n_PCS], 
            %                       projected sampling locations to plot
            % network_type: (str), 'swg' or 'ctxt' 
            
            % constants
            [n_inpConds, n_sampling_locs_per_conds, ~] = size(proj_sampling_locs);
            
            for inpCond_nr = 1:n_inpConds
                for start_pt_nr = 1:n_sampling_locs_per_conds
                    proj_Pt = squeeze(proj_sampling_locs(inpCond_nr, start_pt_nr, :));
                    if strcmp(network_type, 'swg')
                        scatter3(ax, proj_Pt(1), proj_Pt(2), proj_Pt(3), 100, 'k', ...  
                        'filled', 'd', 'HandleVisibility', 'off'); hold on;
                    elseif strcmp(network_type, 'ctxt')
                        scatter(ax, proj_Pt(1), proj_Pt(2), 100, 'k', ... 
                        'filled', 'd', 'HandleVisibility', 'off'); hold on;
                    end    
                end
            end
        end
        

        function [fig, ax] = plot_sampling_locs_on_condAvgTrajs(obj, network_activity, sampling_locs, sampling_loc_props, ...
                                                                all_freq_ids, conditionIds, coherencies_trial, network_type)
            % plot low-dimensional projection of condition average trajectories with added sampling locations 
            % network_activity: [n_units, n_timesteps, n_trials]
            % sampling_loc: [n_input_conditions, n_units, n_start_pts_per_inpCond]
            % sampling_loc_props: (str) with properties of all sampling locations
            % all_freq_ids: [n_trials, 1], all frequency IDs of each trial
            % conditionIds = [1, n_trials], context ID per trial
            % coherencies_trial = [nIntegrators, n_trials], input coherencies of sensory input 1 and 2 over trials      
            % network_type: (str), 'swg' or 'ctxt' 
            
            % constants
            if strcmp(network_type, 'swg')
                nPCs = 3;
            elseif strcmp(network_type, 'ctxt')
                nPCs = 2;
            else
                assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")
            end
            
            % get projections
            PCs                = obj.get_PCs_of_network_activity(network_activity);
            proj_sampling_locs = obj.project_sampling_locs_onto_PCs(sampling_locs, PCs(:,1:nPCs));
            proj_condAvgTrajs  = obj.project_condition_average_trajectories_onto_PCs(network_activity, ...
                                    PCs(:, 1:nPCs), network_type, sampling_loc_props, all_freq_ids, ...
                                    conditionIds, coherencies_trial);
            
            % constants         
            n_inpConds = size(proj_sampling_locs, 1); 
            color_per_inpCond = parula(n_inpConds);

            fig = figure('Position', [10 10 1700 1400]); ax = subplot(1,1,1); grid on; 
            % plot condition average trajectories    
            obj.plot_condAvgTrajs(ax, proj_condAvgTrajs, color_per_inpCond, network_type, 1, '-')
            % plot sampling locations
            obj.plot_sampling_locs(ax, proj_sampling_locs, network_type)
            axis square
            title(ax, "sampling locations (\diamondsuit) on condition average trajectories");
            xlabel(ax, "PC(X)_1"); ylabel(ax, "PC(X)_2"); zlabel(ax, "PC(X)_3");
            ax.FontSize = obj.my_fontsize-15;
        end
        
        
        function [fig, ax] = plot_full_and_reduced_rank_condAvgTrajs(obj, network_activity_fr, network_activity_rr, ...
                                    sampling_loc_props, all_freq_ids, conditionIds, coherencies_trial, network_type, rankW)
            % plot low-dimensional projection of condition average trajectories for full-rank (solid line=) and reduced-rank (dotted line) networks
            % network_activity_fr: [n_units, n_timesteps, n_trials], network activities of reduced-rank network
            % network_activity_rr: [n_units, n_timesteps, n_trials], network activities of reduced-rank network            
            % sampling_loc_props: (str) with properties of all sampling locations
            % all_freq_ids: [n_trials, 1], all frequency IDs of each trial
            % conditionIds = [1,n_trials], context ID per trial
            % coherencies_trial = [nIntegrators, n_trials], input coherencies of sensory input 1 and 2 over trials      
            % network_type: (str), 'swg' or 'ctxt' 
            
            % constants
            if strcmp(network_type, 'swg')
                nPCs = 3;
                n_inpConds = numel(sampling_loc_props.freq_idx_per_inpCond); 
            elseif strcmp(network_type, 'ctxt')
                nPCs = 2;
                n_inpConds = numel(sampling_loc_props.ctxt_per_inpCond); 
            else
                assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")
            end
            color_per_inpCond = parula(n_inpConds);
                       
            % get projections
            PCs                  = obj.get_PCs_of_network_activity(network_activity_fr);
            proj_condAvgTrajs_fr = obj.project_condition_average_trajectories_onto_PCs(network_activity_fr, ...
                                    PCs(:, 1:nPCs), network_type, sampling_loc_props, all_freq_ids, conditionIds, coherencies_trial);
            proj_condAvgTrajs_rr = obj.project_condition_average_trajectories_onto_PCs(network_activity_rr, ...
                                    PCs(:, 1:nPCs), network_type, sampling_loc_props, all_freq_ids, conditionIds, coherencies_trial);
                                
            % plot condition average trajectories    
            fig = figure('Position', [10 10 1800 1400]); ax = subplot(1,1,1); grid on;  % hold on;
            obj.plot_condAvgTrajs(ax, proj_condAvgTrajs_fr, color_per_inpCond, network_type, 1, '-')
            obj.plot_condAvgTrajs(ax, proj_condAvgTrajs_rr, color_per_inpCond*0.7, network_type, 0, ':')            
            title(ax, ["condition average trajectories", strcat("(solid lines: W; dotted lines: W^{PC}_{k=",num2str(rankW), "})")], ...
                    'FontSize', obj.my_fontsize);
            xlabel(ax, "PC(X)_1", 'FontSize', obj.my_fontsize);
            ylabel(ax, "PC(X)_2", 'FontSize', obj.my_fontsize); 
            zlabel(ax, "PC(X)_3", 'FontSize', obj.my_fontsize);
            axis square
            ax.FontSize = obj.my_fontsize-15;            
            
        end
    end
end


