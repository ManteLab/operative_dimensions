classdef UtilsIO
    % class to collect helper functions to deal import/export
    
    methods
        function [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = load_weights(obj, path_to_weights, net_id)
            % load network weights 
            % path_to_weights: (str), path to hdf5-file with stored weight matrices
            % net_id: (integer), number of network in hdf5-file to load
            
            % n_Wru_v: [n_units, n_inputs], input weights
            % n_Wrr_n: [n_units, n_units], recurrent weights
            % m_Wzr_n: [n_outputs, n_inputs], output weights
            % n_x0_c: [n_units, n_contexts], initial conditions per context
            % n_bx_1: [n_units, 1], bias of hidden units
            % m_bz_1: [n_outputs, 1], bias of output units
            
            name_dataset = char(['/NetNr' num2str(net_id) '/final']);

            n_Wru_v = hdf5read(path_to_weights, [name_dataset, '/n_Wru_v']);
            n_Wrr_n = hdf5read(path_to_weights, [name_dataset, '/n_Wrr_n']);
            m_Wzr_n = hdf5read(path_to_weights, [name_dataset, '/m_Wzr_n']);
            n_x0_c  = hdf5read(path_to_weights, [name_dataset, '/n_x0_c']);
            n_bx_1  = hdf5read(path_to_weights, [name_dataset, '/n_bx_1']);
            m_bz_1  = hdf5read(path_to_weights, [name_dataset, '/m_bz_1']);
        end

        function save_to_hdf5(obj, outputfilename, group_name, my_data)
            % save all fields of data structure my_data to hdf5
            % outputfilename: (str), name of hdf5-file and path to it
            % group_name: (str), name of hdf5-group
            % my_data: (structure), contains data to store in fields
            
            all_data_names = fields(my_data);
            for data_nr = 1:numel(all_data_names)
                data_name = all_data_names{data_nr};
                data_values = getfield(my_data, data_name);
                h5create(outputfilename, strcat('/', group_name, '/', data_name), size(data_values), ...
                            'ChunkSize', size(data_values), 'Deflate', 9)
                h5write(outputfilename, strcat('/', group_name, '/', data_name), data_values)
            end
        end
        
    end
end
