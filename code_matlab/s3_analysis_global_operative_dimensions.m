%% imports
base_dir_code = '../';
base_dir_data = '../data/';
set_paths(base_dir_code)

UIO  = utils.UtilsIO;
UPlt = utils.UtilsPlotting;
UOD  = utils.UtilsOpDims;
USL  = utils.UtilsSamplingLocs;

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
    
elseif strcmp(network_type, 'ctxt')
    n_trials = 5;  % trials per context and per input coherency (total = n_trials * 2 * 6)
    
    InpGenCtxt = utils.input_generation.InputGeneratorCtxt;
    [coherencies_trial, conditionIds, inputs, targets] = InpGenCtxt.get_ctxt_dep_integrator_inputOutputDataset(n_trials, with_inputnoise);
    
else
    assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")

end   

%% DIMENSIONALITY OF GLOBAL OPERATIVE DIMENSIONS
% load  local operative dimensions
[sampling_loc_props] = USL.get_sampling_location_properties(network_type);

inputfilename = strcat(base_dir_data, 'local_operative_dimensions/localOpDims_', network_type, '_', dim_type, '.h5');
[all_local_op_dims, all_fvals] = UOD.load_local_op_dims(inputfilename, n_units, sampling_loc_props, network_type);

% combine local operative dimensions to obtain global operative dimensions 
sampling_locs_to_combine = 'all'; % options for ctxt network: 'ctxt1' 'ctxt2' 'allPosChoice' 'allNegChoice'
[global_op_dims, singular_values_of_global_op_dims] = UOD.get_global_operative_dimensions(sampling_locs_to_combine, ...
                                                        sampling_loc_props, all_local_op_dims, all_fvals);

% plot dimensionality of global operative dimensions
var_of_global_op_dims = singular_values_of_global_op_dims.^2;
[fig, ax] = UPlt.plot_lineplot(1:n_units, diag(var_of_global_op_dims)/sum(var_of_global_op_dims, 'all')*100,...
              "Dimensionality of global operative dimensions", "PC(L)_i", "variance explained (%)");
    
%% PERFORMANCE OVER SEQUENTIALLY REMOVING GLOBAL OPERATIVE DIMENSIONS FROM W
% run full-rank network as reference to calculate state distance measure
[forwardPass_org] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, ...
                        n_bx_1, m_bz_1, inputs, conditionIds, seed_run, net_noise);

% run network with reduced-rank W and collect performance measures
% ( = mean squared error (mse) & State distance between full-rank and reduced-rank network trajectories)
n_op_dims         = n_units;
mses              = NaN([n_op_dims, 1]);
statedists_to_org = NaN([n_op_dims, 1]);
for dim_nr = 1:n_op_dims
    % modify W
    n_Wrr_n_modified = utils.remove_dimension_from_weight_matrix(n_Wrr_n, ...
                        global_op_dims(:,dim_nr+1:n_units), dim_type);

    % run modified network
    [forwardPass] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n_modified, ...
                        m_Wzr_n, n_x0_c, n_bx_1, m_bz_1, inputs, conditionIds, ...
                        seed_run, net_noise);

    % get performance measures
    mses(dim_nr, 1) = utils.get_mse(forwardPass.m_z_t, targets, 'all');
    statedists_to_org(dim_nr, 1) = utils.get_state_distance_between_trajs(forwardPass.n_x_t, ...
                                    forwardPass_org.n_x_t);

end

% plot performance over reduced-rank Ws
[fig, ax] = UPlt.plot_lineplot(1:n_units, mses, "network output cost for reduced-rank W", "rank(W^{OP}_k)", "cost");
[fig, ax] = UPlt.plot_lineplot(1:n_units, statedists_to_org, "state distance to trajectory of full-rank W", ...
                "rank(W^{OP}_k)", "state distance (a.u.)");
