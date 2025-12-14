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

%% SHARED CONSTANTS / PARAMETERS
% load network weights
path_to_weights = [base_dir_data, 'pretrained_networks/', network_type, '/', network_type, '_weights.h5'];
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = UIO.load_weights(path_to_weights, net_id);  
n_units = size(n_Wrr_n, 1);
n_inputs = size(n_Wru_v, 2);

% generate network inputs to run network
with_netnoise    = 1;  % 0 or 1
with_inputnoise  = 1;  % 0 or 1, only for ctxt network

seed_run   = 1001;
seed_input = 1000;

% network noise
if with_netnoise
    net_noise   = 'default';
else
    net_noise   = 0;    
end

rng(seed_input);
if strcmp(network_type, 'swg')
    n_trials = 51*10;  % total, over all 51 frequencies
    
    InpGenSW = utils.input_generation.InputGeneratorSWG;
    [all_freqs, all_freq_ids, inputs, targets, conditionIds] = InpGenSW.get_sine_wave_generator_inputOutputDataset(n_trials);
    
elseif strcmp(network_type, 'ctxt')
    n_trials = 10;  % trials per context and per input coherency
    
    InpGenCtxt = utils.input_generation.InputGeneratorCtxt;
    [coherencies_trial, conditionIds, inputs, targets] = InpGenCtxt.get_ctxt_dep_integrator_inputOutputDataset(n_trials, with_inputnoise);
    
else
    assert(false, "Network type unknown, please set network_type to 'swg' or 'ctxt'")

end   


%% DEFINE SAMPLING LOCATIONS ON CONDITION AVERAGE TRAJECTORY
% collect sampling locations for local operative dimensions
% (equally spaced in time, along all condition average trajectories)
[sampling_loc_props] = USL.get_sampling_location_properties(network_type);

% run network to get condition average trajectories
[forwardPass] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1, ...
                                            inputs, conditionIds, seed_run, net_noise);
                
% extract sampling locations based on defined input conditions and temporal sampling,
% then visualize sampling locations 
if strcmp(network_type, 'swg')
    [sampling_locs] = USL.get_sampling_locs_on_condAvgTrajs_swg(forwardPass.n_x_t, ...
        sampling_loc_props, all_freq_ids);
    [fig, ax] = UPlt.plot_sampling_locs_on_condAvgTrajs(forwardPass.n_x_t, sampling_locs, ...
        sampling_loc_props, all_freq_ids, [], [], network_type);                            
                
elseif strcmp(network_type, 'ctxt')
    [sampling_locs] = USL.get_sampling_locs_on_condAvgTrajs_ctxt(forwardPass.n_x_t, ...
        sampling_loc_props, conditionIds, coherencies_trial);
    [fig, ax] = UPlt.plot_sampling_locs_on_condAvgTrajs(forwardPass.n_x_t, sampling_locs, ...
        sampling_loc_props, [], conditionIds, coherencies_trial, network_type);                            
end


%% FIND LOCAL OPERATIVE DIMENSIONS AT EVERY SAMPLING LOCATION
time_stamp = datestr(now,'yyyymmdd_HHMMSS');
outputfilename = strcat(base_dir_data, 'local_operative_dimensions/localOpDims_', network_type, '_', dim_type, '_', time_stamp, '.h5');

thr_fval = 1e-8;
n_dims_to_find  = n_units;
n_sampling_locs = numel(sampling_loc_props.t_start_pt_per_loc);
for loc_nr = 1:n_sampling_locs
    samplingLocParams = [];
    samplingLocParams.local_op_dims = nan([n_dims_to_find, n_units]);
    samplingLocParams.all_fvals     = nan([n_dims_to_find, 1]);

    % get sampling location
    start_pt_nr = USL.map_optLocNr_to_startPtNr(loc_nr, sampling_loc_props);
    inpCond_nr  = USL.map_optLocNr_to_inpCondNr(loc_nr, sampling_loc_props, network_type);
    samplingLocParams.sampling_loc  = squeeze(sampling_locs(inpCond_nr, :, start_pt_nr));
    
    % info on sampling location
    samplingLocParams.t_start_point = sampling_loc_props.t_start_pt_per_loc(loc_nr);
    if strcmp(network_type, 'swg')
        samplingLocParams.freq_id = sampling_loc_props.freq_idx_per_loc(loc_nr);
        [dim_name] = UOD.get_name_of_local_operative_dims(network_type, ...
                            samplingLocParams.t_start_point, [], [], [], samplingLocParams.freq_id);

    elseif strcmp(network_type, 'ctxt')
        samplingLocParams.ctxt_id  = sampling_loc_props.ctxt_per_loc(loc_nr);
        samplingLocParams.signCoh1 = sampling_loc_props.signCoh1_per_loc(loc_nr);
        samplingLocParams.signCoh2 = sampling_loc_props.signCoh2_per_loc(loc_nr);
        [dim_name] = UOD.get_name_of_local_operative_dims(network_type, ...
                        samplingLocParams.t_start_point(1), samplingLocParams.ctxt_id, ...
                        samplingLocParams.signCoh1, samplingLocParams.signCoh2, []);
    end

    % collect trajs for full-rank network as reference
    samplingLocParams.all_trajs_org = nan([n_units, 2]);
    inputs_relax = zeros([n_inputs, 1, 1]);
    if strcmp(network_type, 'swg')
        conditionIds_relax = [1];
        init_n_x0_c = [samplingLocParams.sampling_loc].';
    elseif strcmp(network_type, 'ctxt')
        ctxt_id = samplingLocParams.ctxt_id;
        inputs_relax(2+ctxt_id, :, :) = 0;
        conditionIds_relax = [ctxt_id ctxt_id];
        init_n_x0_c = [samplingLocParams.sampling_loc; samplingLocParams.sampling_loc].';
    end
    net_noise_trajs = 0;
    [forwardPass_modified] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, ...
                                    init_n_x0_c, n_bx_1, m_bz_1, inputs_relax, ...
                                    conditionIds_relax, seed_run, net_noise_trajs);

    % add first step separately and then all the other steps
    samplingLocParams.all_trajs_org(:, 1) = forwardPass_modified.n_x0_1(:, 1);
    samplingLocParams.all_trajs_org(:, 2) = forwardPass_modified.n_x_t;

    if strcmp(dim_type, 'columns')
        samplingLocParams.local_op_dims(1, :) = utils.make_unit_length(n_Wrr_n * tanh(samplingLocParams.sampling_loc).');

        dims_to_be_orth = [];
        fval = utils.get_neg_deltaFF(samplingLocParams.local_op_dims(1, :).', ...
                    dims_to_be_orth, samplingLocParams, n_Wru_v, n_Wrr_n, ...
                    m_Wzr_n, n_bx_1, m_bz_1, dim_type, network_type);
        samplingLocParams.all_fvals = nan([n_units, 1]);
        samplingLocParams.all_fvals(1,1) = fval;

    elseif strcmp(dim_type, 'rows')
        % start optimization process
        for dim_nr = 1:n_dims_to_find
            op_dims = samplingLocParams.local_op_dims(1:dim_nr-1, :).';

            % get random initial conditions
            x0 = utils.make_unit_length(randn([n_units, 1]));            

            % optimize delta f, to find local operative dimension
            dims_to_be_orth = op_dims;
            fun = @(x0)utils.get_neg_deltaFF(x0, dims_to_be_orth, samplingLocParams, ...
                          n_Wru_v, n_Wrr_n, m_Wzr_n, n_bx_1, m_bz_1, dim_type, network_type);
            options = optimset('Display','iter', 'TolX', 1e-6);  % 1e-6);
            [x0, fval] = fminunc(fun, x0, options);

            % postprocess found solution x0 (orth to prev dims, unit length) &
            % ...add new x0 & fval to collection over samples
            [Q, R] = qr([op_dims, utils.make_unit_length(x0)], 0);
            samplingLocParams.local_op_dims(dim_nr, :) = utils.make_unit_length(Q(:, end));
            samplingLocParams.all_fvals(dim_nr, 1) = fval;

            if abs(fval) < thr_fval
                break
            end
        end
    else
        assert(false, "dim_type unknown, please set network_type to 'columns' or 'rows'")
    end

    % save to hdf5
    hdf5_group_name = dim_name;
    UIO.save_to_hdf5(outputfilename, hdf5_group_name, samplingLocParams)
end
