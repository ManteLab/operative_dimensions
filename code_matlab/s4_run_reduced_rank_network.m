%% imports
base_dir_code = '../';
base_dir_data = '../data/';
set_paths(base_dir_code)

UIO  = utils.UtilsIO;
UPlt = utils.UtilsPlotting;
USL  = utils.UtilsSamplingLocs;
UOD  = utils.UtilsOpDims;

% network settings
net_id = 1;
network_type = 'ctxt';  % 'swg': sine wave generator network; 'ctxt': context-dependent integration network
dim_type = 'columns';  % 'columns' 'rows'

% noise settings
with_netnoise    = 1;  % 0 or 1
with_inputnoise  = 1;  % 0 or 1, only for ctxt network

seed_run   = 1001;
seed_input = 1000;

%% SHARED CONSTANTS / PARAMETERS
% load network weights
path_to_weights = [base_dir_data, 'pretrained_networks/', network_type, '/', network_type, '_weights.h5'];
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = UIO.load_weights(path_to_weights, net_id);  
n_units = size(n_Wrr_n, 1);

% network noise
if with_netnoise
    net_noise   = 'default';
else
    net_noise   = 0;
end

% generate network inputs to run network
rng(seed_input);
if strcmp(network_type, 'swg')
    n_trials = 51*5;  % total, over all 51 frequencies
    InpGenSW = utils.input_generation.InputGeneratorSWG;
    [all_freqs, all_freq_ids, inputs, targets, conditionIds] = InpGenSW.get_sine_wave_generator_inputOutputDataset(n_trials);
    coherencies_trial = []; 
    
elseif strcmp(network_type, 'ctxt')
    n_trials = 5;  % trials per context and per input coherency
    InpGenCtxt = utils.input_generation.InputGeneratorCtxt;
    [coherencies_trial, conditionIds, inputs, targets] = InpGenCtxt.get_ctxt_dep_integrator_inputOutputDataset(n_trials, with_inputnoise);
    all_freq_ids = [];
    
else
    assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")

end   


%% GENERATE GLOBAL OPERATIVE DIMENSIONS
% load  local operative dimensions
[sampling_loc_props] = USL.get_sampling_location_properties(network_type);

inputfilename = strcat(base_dir_data, 'local_operative_dimensions/localOpDims_', network_type, '_', dim_type, '.h5');
[all_local_op_dims, all_fvals] = UOD.load_local_op_dims(inputfilename, n_units, sampling_loc_props, network_type);

% combine local operative dimensions to obtain global operative dimensions 
sampling_locs_to_combine = 'all'; % options for ctxt network: 'ctxt1' 'ctxt2' 'allPosChoice' 'allNegChoice'
[global_op_dims, singular_values_of_global_op_dims] = UOD.get_global_operative_dimensions(sampling_locs_to_combine, ...
                                                        sampling_loc_props, all_local_op_dims, all_fvals);

%% RUN FULL-RANK AND REDUCED-RANK NETWORK
% & COMPARE NETWORK OUTPUT AND CONDITION AVERAGE TRAJECTORIES
if strcmp(dim_type, 'columns')
    if strcmp(network_type, 'ctxt')
        rankW = 15;  % set rank of reduced-rank W
    elseif strcmp(network_type, 'swg')
        rankW = 29;
    end
elseif strcmp(dim_type, 'rows')
    if strcmp(network_type, 'ctxt')
        rankW = 27;  % set rank of reduced-rank W
    elseif strcmp(network_type, 'swg')
        rankW = 41;
    end
end

% run full-rank network
[forwardPass_fullRank] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, ...
                            n_x0_c, n_bx_1, m_bz_1, inputs, conditionIds, seed_run, net_noise);
% run reduced-rank network
n_Wrr_n_modified = utils.remove_dimension_from_weight_matrix(n_Wrr_n, ...
                    global_op_dims(:,rankW+1:n_units), dim_type);
[forwardPass_reducedRank] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n_modified, ...
                                m_Wzr_n, n_x0_c, n_bx_1, m_bz_1, ...
                                inputs, conditionIds, seed_run, net_noise);

% plot network outputs for several trials
for trial_nr = 1:5:26
    [fig, ax] = UPlt.plot_lineplot(1:size(targets, 2), ...
        [targets(1, :, trial_nr); forwardPass_fullRank.m_z_t(1, :, trial_nr); forwardPass_reducedRank.m_z_t(1, :, trial_nr)], ...
        "network output", "t", "z_t", 'display_names', ["target" "full-rank" "reduced-rank"]);
    ylim(ax, [-1.1 1.1]); hold on;
end
                        
% plot one example trials for network trajectory
[fig, ax] = UPlt.plot_full_and_reduced_rank_condAvgTrajs(forwardPass_fullRank.n_x_t, ...
                forwardPass_reducedRank.n_x_t, sampling_loc_props, all_freq_ids, ...
                conditionIds, coherencies_trial, network_type, rankW);



