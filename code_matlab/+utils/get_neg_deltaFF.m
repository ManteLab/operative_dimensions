function [neg_deltaFF] = get_neg_deltaFF(x0, dims_to_be_orth, samplingLocParams, ...
                                            n_Wru_v, n_Wrr_n, m_Wzr_n, n_bx_1, m_bz_1, dim_type, network_type)
    % calculate impact of removing dimension x0 from W as euclidean
    % distance between x_t and ^x_t.
    
    % x0: [n_units, 1], dimension to remove from W
    % dims_to_be_orth: [n_units, n_dims], set of dimensions to which x0 has to be orthogonal
    % samplingLocParams: [structure], parameters of sampling location which is currently tested  
    % n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1: network weights
    % dim_type: (str), 'columns' or 'rows', decide which dimension type should be removed from W
    % network_type: (str), 'swg' or 'ctxt'

    n_units  = numel(x0);
    n_inputs = size(n_Wru_v, 2);
    
    % make x0 ortogonal to prev found dimensions and unit length
    x0 = utils.make_unit_length(x0);
    [Q, R] = qr([dims_to_be_orth, x0], 0);
    x0 = Q(:, end);
    n_Wrr_n_modified = utils.remove_dimension_from_weight_matrix(n_Wrr_n, x0, dim_type);               

    % run network at sampling location and collect trajectories
    inputs_relax = zeros([n_inputs, 1, 1]);
    if strcmp(network_type, 'swg')
        conditionIds_relax = [1];
        init_n_x0_c = [samplingLocParams.sampling_loc].';
    elseif strcmp(network_type, 'ctxt')
        ctxt_id      = samplingLocParams.ctxt_id;
        inputs_relax(2+ctxt_id, :, :) = 0;
        conditionIds_relax = [ctxt_id ctxt_id];
        init_n_x0_c = [samplingLocParams.sampling_loc; samplingLocParams.sampling_loc].';
    else
        assert(false)
    end   
    net_noise_trajs = 0;
    [forwardPass_modified] = utils.run_one_forwardPass(n_Wru_v, n_Wrr_n_modified, ...
                                            m_Wzr_n, init_n_x0_c, n_bx_1, m_bz_1, ...
                                            inputs_relax, conditionIds_relax, [], net_noise_trajs);

    % add first step separately and then all the other steps
    all_trajs = nan([n_units, 2]);
    all_trajs(:, 1) = forwardPass_modified.n_x0_1(:, 1);
    all_trajs(:, 2) = forwardPass_modified.n_x_t;
    
    % collect state distance between trajectories
    trajs_orgWrr = squeeze(samplingLocParams.all_trajs_org(: , 2:end));
    trajs_modWrr = squeeze(all_trajs(: , 2:end));
    state_dist_to_org_net= squeeze(utils.get_state_distance_between_trajs(trajs_orgWrr, trajs_modWrr));
    
    neg_deltaFF = state_dist_to_org_net * -1;  % inverse to find maxima with fminunc method... 
    
end
