classdef UtilsOpDims
    % class to collect helper functions to deal with operative dimensions
    
    methods
        function [all_local_op_dims, all_fvals] = load_local_op_dims(obj, inputfilename, n_units, ...
                                                    sampling_loc_props, network_type)        
            % load local operative dimensions from file
            % inputfilename: (str), path to hdf5-file with local opeartive dimensions
            % n_units: (int), number of hidden units
            % sampling_loc_props: (str) with properties of all sampling locations
            % network_type: (str), 'swg' or 'ctxt'

            % all_local_op_dims: [n_locs, n_op_dims, n_units], all local operative dimensions at sampling locations
            % all_fvals: [n_locs, n_op_dims], delta f for all local operative dimensions at all sampling locations 
                                                
            % constants
            n_dims_to_load = n_units;                                                              
            n_locs = numel(sampling_loc_props.t_start_pt_per_loc);
            
            % load 
            all_local_op_dims = nan([n_locs, n_dims_to_load, n_units]);
            all_fvals          = nan([n_locs, n_dims_to_load]);
            for loc_nr = 1:n_locs
                % find dimension name
                if strcmp(network_type, 'swg')
                    freq_id = sampling_loc_props.freq_idx_per_loc(loc_nr);
                    ctxt_id  = []; 
                    signCoh1 = [];
                    signCoh2 = [];
                elseif strcmp(network_type, 'ctxt')
                    ctxt_id  = sampling_loc_props.ctxt_per_loc(loc_nr);
                    signCoh1 = sampling_loc_props.signCoh1_per_loc(loc_nr);
                    signCoh2 = sampling_loc_props.signCoh2_per_loc(loc_nr);
                    freq_id  = [];
                else
                    assert(false)
                end
                t_start_pt = sampling_loc_props.t_start_pt_per_loc(loc_nr);
                [dim_name] = obj.get_name_of_local_operative_dims(network_type, ...
                                t_start_pt, ctxt_id, signCoh1, signCoh2, freq_id);

                % load
                fvals         = hdf5read(inputfilename, strcat(dim_name, '/all_fvals'));
                local_op_dims = hdf5read(inputfilename, strcat(dim_name, '/local_op_dims'));
                
                % ensure correct formating
                for dim_nr = 1:n_dims_to_load
                    all_local_op_dims(loc_nr, dim_nr, :) = squeeze(local_op_dims(dim_nr, :));
                    all_fvals(loc_nr, dim_nr) = fvals(dim_nr);
                end
            end
        end
        
        
        function [dim_name] = get_name_of_local_operative_dims(obj, network_type, ...
                                t_start_point, ctxt_id,  signCoh1, signCoh2, freq_id)
            % get name of local operative dimension (dim_name) based on properties of
            % sampling location
            % network_type: (str), 'swg' or 'ctxt'
            % t_start_pt_per_loc: (int), t along trial in condition average trajectory
            % ctxt_id: (int), context ID ([] for swg network)
            % signCoh1: (int), sign of sensory input 1 ([] for swg network)
            % signCoh2: (int), sign of sensory input 2 ([] for swg network)
            % freq_id: (int), frequency ID ([] for ctxt network)
            
            % dim_name: (str), human interpretable name of current sampling location
            
            if strcmp(network_type, 'swg')
                dim_name = strcat('opt_dims_t', num2str(t_start_point), 'Freq', num2str(freq_id));    
            
            elseif strcmp(network_type, 'ctxt')
                dim_name = strcat('opt_dims_t', num2str(t_start_point), 'Ctxt', num2str(ctxt_id), ...
                            'Inps', num2str(signCoh1), '_', num2str(signCoh2));    
            
            else 
                assert(false)
            end
            
        end
                
        
        function [all_lSV, all_SVals] = get_global_operative_dimensions(obj, sampling_locs_to_combine, ...
                                                    sampling_loc_props, all_local_op_dims, all_fvals_dims)
            % combine local operative dimension into global operative dimensions
            
            % sampling_locs_to_combine: (str), defines which subset of
            %       local operative dimensions should be considered to generate
            %       the global operative dimensions (see explanations on function-specific
            %       dimensions in paper)
            %       currently only implemented for ctxt-network
            % sampling_loc_props: (str) with properties of all sampling locations
            % all_local_op_dims: [n_locs, n_op_dims, n_units], all local operative dimensions at sampling locations
            % all_fvals: [n_locs, n_op_dims], delta f for all local operative dimensions at all sampling locations 

            % constants
            n_units = size(all_local_op_dims, 3);
            n_locs_total = size(all_fvals_dims, 1);
            
            % define which sampling locations to consider
            if strcmp(sampling_locs_to_combine, 'all')
                loc_nrs = 1:n_locs_total;
            elseif strcmp(sampling_locs_to_combine, 'ctxt1')
                loc_nrs = find(sampling_loc_props.ctxt_per_loc==1);
            elseif strcmp(sampling_locs_to_combine, 'ctxt2')
                loc_nrs = find(sampling_loc_props.ctxt_per_loc==2);                    
            elseif strcmp(sampling_locs_to_combine, 'allPosChoice')
                loc_nrs = find(((sampling_loc_props.ctxt_per_loc==1) & (sampling_loc_props.signCoh1_per_loc>0)) | ...
                    ((sampling_loc_props.ctxt_per_loc==2) & (sampling_loc_props.signCoh2_per_loc>0)));
            elseif strcmp(sampling_locs_to_combine, 'allNegChoice')
                loc_nrs = find(((sampling_loc_props.ctxt_per_loc==1) & (sampling_loc_props.signCoh1_per_loc<0)) | ...
                    ((sampling_loc_props.ctxt_per_loc==2) & (sampling_loc_props.signCoh2_per_loc<0)));               
            else
                assert(false, "sampling_locs_to_combine unknown. Please choose a valid option")
            end
            n_locs = numel(loc_nrs);

            % combine all considered sampling locations into one matrix
            % as locOpDim * localDeltaF, ...
            % then SVD(L)
            counter = 0;
            L = zeros([n_units, n_locs*n_units]);
            for loc_nr = loc_nrs
                for dim_nr = 1:n_units
                    counter = counter + 1;
                    if ~isnan(all_fvals_dims(loc_nr, dim_nr))
                        L(:, counter) = squeeze(all_local_op_dims(loc_nr, dim_nr, :)*all_fvals_dims(loc_nr, dim_nr));
                    end 
                end    
            end
            [all_lSV, all_SVals, ~] = svd(L, 'econ');
        end
        
    end
end




            
                
                
                
                
                
                









