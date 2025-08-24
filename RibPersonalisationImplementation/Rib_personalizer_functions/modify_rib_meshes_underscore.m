function modify_rib_meshes_underscore(input_file, output_file)
    % Replace rib meshes with _pers versions
    
    try
        import org.opensim.modeling.*
        
        fprintf('Loading OpenSim model: %s\n', input_file);
        model = Model(input_file);
        
        bodies = model.getBodySet();
        updated_count = 0;
        
        for i = 0:bodies.getSize()-1
            body = bodies.get(i);
            geom_set = body.getPropertyByName('attached_geometry');
            
            for j = 0:geom_set.size()-1
                current_geom = geom_set.getValueAsObject(j);
                
                if strcmp(char(current_geom.getConcreteClassName()), 'Mesh')
                    mesh = Mesh.safeDownCast(current_geom);
                    current_file = char(mesh.get_mesh_file());
                    
                    if contains(current_file, 'Rib') && ~contains(current_file, '_pers')
                        [~, base_name, ~] = fileparts(current_file);
                        new_file = [base_name '_pers.obj'];
                        mesh.set_mesh_file(new_file);
                        
                        fprintf('Updated mesh %s to %s\n', current_file, new_file);
                        updated_count = updated_count + 1;
                    end
                end
            end
        end
        
        model.print(output_file);
        fprintf('Successfully updated %d rib meshes\n', updated_count);
        
    catch ME
        fprintf('Error: %s\n', ME.message);
        rethrow(ME);
    end
end