%% imports
base_dir_code = '../';
base_dir_data = '../data/';
set_paths(base_dir_code)

UIO  = utils.UtilsIO;
UPlt = utils.UtilsPlotting;

% network settings
net_id = 1;
network_type = 'ctxt';  % 'swg': sine wave generator network; 'ctxt': context-dependent integration network

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
    n_trials = 51*3;  % total, over all 51 frequencies
    
    InpGenSW = utils.input_generation.InputGeneratorSWG;
    [all_freqs, all_freq_ids, inputs, targets, conditionIds] = InpGenSW.get_sine_wave_generator_inputOutputDataset(n_trials);
    
elseif strcmp(network_type, 'ctxt')
    n_trials = 5;  % trials per context and per input coherency
    
    InpGenCtxt = utils.input_generation.InputGeneratorCtxt;
    [coherencies_trial, conditionIds, inputs, targets] = InpGenCtxt.get_ctxt_dep_integrator_inputOutputDataset(n_trials, with_inputnoise);
    
else
    assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")

end   


%% DIMENSIONALITY OF HIGH-VARIANCE DIMENSIONS OF W
% plot dimensionality of high-variance dimensions (perform SVD(W))
[S] = svd(n_Wrr_n);
S = S.^2;
[fig, ax] = UPlt.plot_lineplot(1:n_units, S/sum(S, 'all')*100, "Dimensionality W", ...
                "PC(W)_i", "variance explained (%)");


%% DIMENSIONALITY OF NETWORK ACTIVITY
% run full-rank network
[forwardPass_fullRank] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, ...
                            n_bx_1, m_bz_1, inputs, conditionIds, seed_run, net_noise);
                
% plot dimensionality of network activities
net_activities = forwardPass_fullRank.n_x_t(:, :);
[~, ~, LATENT] = pca(net_activities.');
[fig, ax] = UPlt.plot_lineplot(1:n_units, LATENT/sum(LATENT, 'all')*100, ...
                "Dimensionality X", "PC(X)_i", "variance explained (%)");


%% PERFORMANCE OVER SEQUENTIALLY REMOVING HIGH-VARIANCE DIMENSIONS FROM W
% get high-variance dimensions
[U, ~, ~] = svd(n_Wrr_n);
   
% run network with reduced-rank W and collect performance measures
% ( = mean squared error (mse) & State distance between full-rank and reduced-rank network trajectories)
n_high_var_dims   = n_units;
mses              = NaN([n_high_var_dims, 1]);
statedists_to_org = NaN([n_high_var_dims, 1]);
for dim_nr = 1:n_high_var_dims
    
    % modify W
    n_Wrr_n_modified = utils.remove_dimension_from_weight_matrix(n_Wrr_n, ...
                        U(:,dim_nr+1:n_units), 'columns');

    % run modified network
    [forwardPass] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n_modified, m_Wzr_n, ...
                    n_x0_c, n_bx_1, m_bz_1, inputs, conditionIds, seed_run, net_noise);

    % get performance measures
    mses(dim_nr, 1) = utils.get_mse(forwardPass.m_z_t, targets, 'all');
    statedists_to_org(dim_nr, 1) = utils.get_state_distance_between_trajs(forwardPass.n_x_t, ...
                                        forwardPass_fullRank.n_x_t);
end

% plot
[fig, ax] = UPlt.plot_lineplot(1:n_units, mses, "network output cost for reduced-rank W", "rank(W^{PC}_k)", "cost");
[fig, ax] = UPlt.plot_lineplot(1:n_units, statedists_to_org, "state distance to trajectory of full-rank W", ...
                "rank(W^{PC}_k)", "state distance (a.u.)");

