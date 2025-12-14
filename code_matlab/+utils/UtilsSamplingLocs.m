classdef UtilsSamplingLocs
    % class to collect helper function to collect sampling locations
    
    methods
        function [sampling_loc_props] = get_sampling_location_properties(obj, network_type)
            % get properties of sampling locations, 
            % here sampling locations defined along condition average
            % trajectories, equally sampled in time (ctxt: 1:100:14000, swg: 1:50:500)            
            % network_type: (str), 'swg' or 'ctxt' 
            
            % sampling_loc_props: (str) with properties of all sampling locations
            % t_start_pt_per_loc: (list), t along trial in condition average trajectory of all sampling locations
            % freq_idx_per_loc: (list), frequency ID of all sampling locations ([] for ctxt network)
            % ctxt_per_loc: (list), context ID all sampling locations ([] for swg network)
            % signCoh1_per_loc: (list), sign of sensory input 1 of all sampling locations ([] for swg network)
            % signCoh2_per_loc: (list), sign of sensory input 2 of all sampling locations ([] for swg network)
            % t_sampling_locs_per_cond: (list), t along trial in condition average trajectory per input conditions
            % freq_idx_per_inpCond: (list), frequency ID per input conditions ([] for ctxt network)
            % ctxt_per_inpCond: (list), context ID per input conditions ([] for swg network)
            % signCoh1_per_inpCond (list), sign of sensory input 1 per input conditions ([] for swg network)
            % signCoh2_per_inpCond: (list), sign of sensory input 2 per input conditions ([] for swg network)
            % inpCond_names: (cell), human-readable names of input conditions 
                
            % default values
            sampling_loc_props = []; 

            if strcmp(network_type, 'swg')
                % every 50-th time step, every 5-th input frequency ID (1:5:51)
                sampling_loc_props.inpCond_names           = {"Freq1" "Freq6" "Freq11" "Freq16" "Freq21" "Freq26" "Freq31" "Freq36" "Freq41" "Freq46" "Freq51"};
                sampling_loc_props.freq_idx_per_inpCond    = [1:5:51];                             
                n_inpConds              = numel(sampling_loc_props.freq_idx_per_inpCond);    
                sampling_loc_props.t_sampling_locs_per_cond = [1 50 100 150 200 250 300 350 400 450 500];
                nT_per_cond             = numel(sampling_loc_props.t_sampling_locs_per_cond);
                
                sampling_loc_props.t_start_pt_per_loc = repmat(sampling_loc_props.t_sampling_locs_per_cond, 1, n_inpConds);
                sampling_loc_props.freq_idx_per_loc   = [ones([1, nT_per_cond])*1  ones([1, nT_per_cond])*6  ones([1, nT_per_cond])*11 ...
                                                          ones([1, nT_per_cond])*16 ones([1, nT_per_cond])*21 ones([1, nT_per_cond])*26 ...
                                                          ones([1, nT_per_cond])*31 ones([1, nT_per_cond])*36 ones([1, nT_per_cond])*41 ...
                                                          ones([1, nT_per_cond])*46 ones([1, nT_per_cond])*51];                

            elseif strcmp(network_type, 'ctxt')
                % every 100-th time step, every input conditions (context 1/2 x pos/neg. choice x coh/incoh. sensory inputs = 8 input conditions)
                sampling_loc_props.inpCond_names             = {"Ctxt1-pos-coh" "Ctxt1-pos-incoh" "Ctxt1-neg-coh" "Ctxt1-neg-incoh" ...
                                                                  "Ctxt2-pos-coh" "Ctxt2-pos-incoh" "Ctxt2-neg-coh" "Ctxt2-neg-incoh"};
                sampling_loc_props.ctxt_per_inpCond          = [1  1  1  1 2  2  2  2];                             
                sampling_loc_props.signCoh1_per_inpCond      = [1  1 -1 -1 1 -1 -1  1];                             
                sampling_loc_props.signCoh2_per_inpCond      = [1 -1 -1  1 1  1 -1 -1]; 
                nInpConds_ctx             = numel(sampling_loc_props.ctxt_per_inpCond);
                sampling_loc_props.t_sampling_locs_per_cond  = [1 100:100:1400];
                nT_per_cond               = numel(sampling_loc_props.t_sampling_locs_per_cond);
                
                sampling_loc_props.t_start_pt_per_loc = [1400 1400 1300 1300 1200 1200 1100 1100 1000 1000  900  900  800  800  700  700  600  600  500  500  400  400  300  300  200  200  100  100   1   1 ...
                                                         1    1  100  100  200  200  300  300  400  400  500  500  600  600  700  700  800  800  900  900 1000 1000 1100 1100 1200 1200 1300 1300 1400 1400 ...
                                                         1400 1400 1300 1300 1200 1200 1100 1100 1000 1000  900  900  800  800  700  700  600  600  500  500  400  400  300  300  200  200  100  100   1   1 ...
                                                         1    1  100  100  200  200  300  300  400  400  500  500  600  600  700  700  800  800  900  900 1000 1000 1100 1100 1200 1200 1300 1300 1400 1400];
                sampling_loc_props.ctxt_per_loc        = [ones([1, nInpConds_ctx/2*nT_per_cond]) ones([1, nInpConds_ctx/2*nT_per_cond])*2];
                sampling_loc_props.signCoh1_per_loc    = [ones([1, nInpConds_ctx/4*nT_per_cond]) ones([1, nInpConds_ctx/4*nT_per_cond])*-1 ...
                                                            repmat([1 -1], 1, nInpConds_ctx/8*nT_per_cond) repmat([-1 1], 1, nInpConds_ctx/8*nT_per_cond)];
                sampling_loc_props.signCoh2_per_loc    = [repmat([1 -1], 1, nInpConds_ctx/8*nT_per_cond) repmat([-1 1], 1, nInpConds_ctx/8*nT_per_cond) ...
                                                            ones([1, nInpConds_ctx/4*nT_per_cond])  ones([1, nInpConds_ctx/4*nT_per_cond])*-1];
                
            else
                assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")
            end
        end
        
    
        function [sampling_locs] = get_sampling_locs_on_condAvgTrajs_ctxt(obj, network_activity, ...
                                    sampling_loc_props, conditionIds, coherencies_trial)
            % find sampling locations as network activitiy vectors from
            % network activity based on defined properties of sampling
            % locations - CONTEXT-DEPENDENT INTEGRATION NETWORK
            
            % network_activity: [n_units, n_timesteps, n_trials], network activity x_t over t and trials
            % sampling_loc_props: (str) with properties of all sampling locations
            % conditionIds = [1, n_trials], context ID per trial
            % coherencies_trial = [nIntegrators, n_trials], input coherencies of sensory input 1 and 2 over trials
            
            % sampling_locs: [n_inpConds, n_units, n_sampling_locs], sampling locations sorted by input conditions
            
            % constants
            n_inpConds = numel(sampling_loc_props.ctxt_per_inpCond);
            n_units = size(network_activity, 1);
            n_timesteps_total = size(network_activity, 2);

            mean_traj_per_inpCond = nan([n_inpConds, n_units, n_timesteps_total]);
            for inpCond_nr = 1:n_inpConds
                valid_trial_ids = conditionIds==sampling_loc_props.ctxt_per_inpCond(inpCond_nr) & ...
                                    (sign(coherencies_trial(1, :))==sampling_loc_props.signCoh1_per_inpCond(inpCond_nr)) & ...
                                    (sign(coherencies_trial(2, :))==sampling_loc_props.signCoh2_per_inpCond(inpCond_nr));
                mean_traj_per_inpCond(inpCond_nr, :, :) = mean(network_activity(:, :, valid_trial_ids), 3);
            end

            % get sampling_locations along mean trajectory
            sampling_locs = mean_traj_per_inpCond(:, :, sampling_loc_props.t_sampling_locs_per_cond);
        end


        function [sampling_locs] = get_sampling_locs_on_condAvgTrajs_swg(obj, network_activity, ...
                                        sampling_loc_props, all_freq_ids)
            % find sampling locations as network activitiy vectors from
            % network activity based on defined properties of sampling
            % locations - SINE WAVE GENERATION NETWORK
            
            % network_activity: [n_units, n_timesteps, n_trials], network activity x_t over t and trials
            % sampling_loc_props: (str) with properties of all sampling locations
            % all_freq_ids: [n_trials, 1], all frequency IDs of each trial
            
            % sampling_locs: [n_inpConds, n_units, n_sampling_locs], sampling locations sorted by input conditions

            % constants
            n_inpConds = size(sampling_loc_props.freq_idx_per_inpCond, 2);
            n_units = size(network_activity, 1);
            n_timesteps_total = size(network_activity, 2);

            mean_traj_per_inpCond = nan([n_inpConds, n_units, n_timesteps_total]);
            for inpCond_nr = 1:n_inpConds
                valid_trial_ids = find(all_freq_ids(:, 1)==sampling_loc_props.freq_idx_per_inpCond(1, inpCond_nr));
                assert(numel(valid_trial_ids)>1, 'not enough trials per inpCond')
                mean_traj_per_inpCond(inpCond_nr, :, :) = mean(network_activity(:, :, valid_trial_ids), 3);
            end

            % get sampling_locations along mean trajectory at t_sampling_locs_per_cond 
            sampling_locs = mean_traj_per_inpCond(:, :, sampling_loc_props.t_sampling_locs_per_cond);
            
        end  
        
        
        
        function [start_pt_nr] = map_optLocNr_to_startPtNr(obj, opt_loc_nr, sampling_loc_props)
            % mapping from 'sampling location number' to 'start_point_number' within on input condition
            % sampling_loc_props: (str) with properties of all sampling locations
            start_pt_nr =  find(sampling_loc_props.t_sampling_locs_per_cond==sampling_loc_props.t_start_pt_per_loc(opt_loc_nr));
            
        end

        
        function [inpCond_nr] = map_optLocNr_to_inpCondNr(obj, opt_loc_nr,sampling_loc_props, network_type)    
            % mapping from 'sampling location number' to 'input condition number' within on input condition
            % sampling_loc_props: (str) with properties of all sampling locations
            if strcmp(network_type, 'swg')
                inpCond_nr = find(sampling_loc_props.freq_idx_per_loc(opt_loc_nr) == sampling_loc_props.freq_idx_per_inpCond);
                
            elseif strcmp(network_type, 'ctxt')
                inpCond_nr = find(sampling_loc_props.ctxt_per_inpCond==sampling_loc_props.ctxt_per_loc(opt_loc_nr) & ...
                                sampling_loc_props.signCoh1_per_inpCond == sampling_loc_props.signCoh1_per_loc(opt_loc_nr) & ...
                                sampling_loc_props.signCoh2_per_inpCond == sampling_loc_props.signCoh2_per_loc(opt_loc_nr));

            else
                assert(false)
                
            end
        end
        
    end
end

