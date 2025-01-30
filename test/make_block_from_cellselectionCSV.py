#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 23:37:06 2025


take a csv from selected cells from paraview
and use these to make a new block for moose mesh refinment

@author: floriankonrad
"""
import sys
import numpy as np
import pandas as pd
import os
import copy
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(script_dir + '/..'))
import exodusio as exio



def make_new_block_from_elementids(block_connects,
                                   ele_per_existing_block_for_new_block,
                                   cell_data):
    '''
    This function is meant for meshes containing only 3D elements
    --> no well path
    --> use before adding wellpath
    It takes a list of lists containing some element ids corresponding to each existing mesh block
    and uses them to generate a new block from it.
    
    It will remove these elements from existing blocks and also adapts cell_data to the new block structure
    and returns an element map to map old element ids to new element ids, eg. for sideset correction

    Parameters
    ----------
    
    block_connects : list of cell arrays
        a list containing an array for each element block
        defining the element connectivity for that block. 
        eg 2 blocks with two and three hex elements (8 node ides per block) will look like this:
            [
                np.array([
                       [117207, 394443, 394444, ..., 394447, 394448, 394449],
                       [394443, 371543, 394450, ..., 394451, 394452, 394448],
                       [394445, 394444, 394453, ..., 394448, 394454, 394455]
                        ]),
                np.array([
                       [491341, 491342, 491348, ..., 383927, 486319, 486317],
                       [491344, 491347, 491349, ..., 486317, 486321, 383929]
                       ])
            ]
        eg use mesh.cells from a meshio mesh object to generate it like this:
        block_connects = []
        for cellblock in mesh.cells:
            block_connects.append(cellblock.data)        
        
    ele_per_existing_block_for_new_block : list of lists
        contains the element ids for each existing block that should be moved to a new block. 
        eg.
        [[1,2],[2]]
        
    cell_data : dict of dicts
        a dict of all data variables in the style of {variable_name = {data_on_blocks}}. 
        data_on_blocks is again a dict where the actual values are stored for block existing
        in the mesh we want to modify. Must correspond in dimensions to block_connects
        the example above has two arrays, first with three second with two elements.
        So cell data must contain for each variable a dict defining this variable with three values on
        first and two values on second block:
        eg. data_on_blocks = {0 : [3,4,3],
                              1: [7,2]}
        eg mesh.cell_data from a meshio mesh object.


    Returns
    -------
    
    new_block_connects : list of arrays
        the new version of the input block_connects
        a list containing an array for each element block including the newly generated,
        which has been appended to the existing and cleaned up blocks,
        Each array is defining the element connectivity for its block. 
        
    new_cell_data : dict of dicts
        the corrected version of the input cell_data incorporating the new block
        a dict of all data variables in the style of {variable_name = {data_on_blocks}}. 
    
    element_map : dict of int32
        element index map, 1-base as per exodus convention
        old_index : new_index
    
    '''
    # remove element_ids_touching_well from their current blocks
    # store those elements (node id definition) 
    new_block_connects = []
    #cells_of_new_block = np.empty((0, 8)) # for hexaeder
    cells_of_new_block = np.empty((0, 4), dtype='int32') # for tetraeder
    old_cell_ids = np.array([], dtype='int32')
    
    for block_id, cellblock in enumerate(block_connects):
        if len(ele_per_existing_block_for_new_block[block_id]) > 0:
            # finding cell definitions that should stay on current block 
            # = elements of current block - selected cells that should be in new block
            corrected_cells = np.delete(cellblock, ele_per_existing_block_for_new_block[block_id], axis=0)
            
            #storing og indices of elems after deletion command
            indices = np.ones(shape=len(cellblock), dtype=bool)
            indices[ele_per_existing_block_for_new_block[block_id]] = False
            og_indices = np.nonzero(indices)[0]
            if block_id > 0:
                # calc how many elems were in all og blocks before the current
                delta_index = sum(len(block) for block in block_connects[:block_id])
                # shift og indices by delta_index to generate global og element indices
                og_indices = og_indices + delta_index
                
            old_cell_ids = np.r_[old_cell_ids,og_indices]
            
            # getting cell definitions of elements that should move to new block
            new_block_cells_on_c_blk = cellblock[ele_per_existing_block_for_new_block[block_id]]
            # appending these cell definitions into collection array to generate new block from
            cells_of_new_block = np.r_[cells_of_new_block,new_block_cells_on_c_blk]
        else:
            # this means no cells of current block should be move to new block
            corrected_cells = cellblock
            
            # og indices array must be filled with information
            og_indices = np.array(range(len(cellblock)), dtype='int32')
            if block_id > 0:
                # calc how many elems were in all blocks before the current
                delta_index = sum(len(block) for block in block_connects[:block_id])
                # shift og indices by delta_index to generate global og element indices
                og_indices = og_indices + delta_index
                
            old_cell_ids = np.r_[old_cell_ids,og_indices]
            
        # generating a new list of block definitions with blocks containing the
        # remaining cells, this is currently without the new block
        new_block_connects.append(corrected_cells)
    
    # adding new block to new list of block definitions
    new_block_connects.append(cells_of_new_block)
    
    # find element map, as moveing elements to a new block changes their element ids
    # which means sidesets and other stuff relying on cell ids wont work anymore
    # therefore generating a cell_map of old cell id and new cell id
    num_elems = sum(len(block) for block in new_block_connects)
    new_cell_ids = np.array(range(num_elems), dtype='int32') + 1
    
    # old cell ids are missing cell ids that movde to new block
    # this information is stored in ele_per_existing_block_for_new_block
    og_indices = np.concatenate(ele_per_existing_block_for_new_block)
    old_cell_ids = np.r_[old_cell_ids,og_indices]
    old_cell_ids = old_cell_ids.astype(np.int32) + 1
    #using stored og indices to generate element map
    element_map = {int(old): int(new) for old, new in zip(old_cell_ids,new_cell_ids)}
    


    # use identified indices reformat cell data as well
    if cell_data is not None:
        new_cell_data = dict()
        for key, arrs in cell_data.items():
            block_data = []
            well_surr_data = np.array([])
            for block_id, data in enumerate(arrs):
                if len(ele_per_existing_block_for_new_block[block_id]) > 0:
                    main_data = np.delete(data, ele_per_existing_block_for_new_block[block_id], axis=0)
                    block_data.append(main_data)
                    well_sur_data_c_block = data[ele_per_existing_block_for_new_block[block_id]]
                    well_surr_data = np.r_[well_surr_data,well_sur_data_c_block]
                else:
                    block_data.append(data)
                    
            block_data.append(well_surr_data)
            new_cell_data[key] = block_data
    else:
        new_cell_data = None
    
    return new_block_connects, new_cell_data, element_map

meshio_type_from_dim = {2: "line",
                        3: "triangle",
                        4: "tetra",
                        8: "hexahedron"}



cell_selections_path = script_dir #"/Users/floriankonrad/Dropbox/5_numys/5_Numys_Projects/03_Kilinkum_Bogenhausen/01_Python/mesh_refining"

df = pd.read_csv(os.path.join(cell_selections_path,"cells_selection_for_ref.csv"))
cell_ids = df["Cell ID"].to_numpy()

ele_per_existing_block_for_new_block = [cell_ids,[],[]]

unrefined_mesh_path = os.path.join(script_dir,"440k_KlinikumBogenhausen_mesh.e") # "/Users/floriankonrad/Dropbox/5_numys/5_Numys_Projects/03_Kilinkum_Bogenhausen/02_Meshing/00_exports/440k_KlinikumBogenhausen_mesh.e"

points,cells,point_data,cell_data,point_sets,info,time_vals,eb_names,side_sets = exio.read(unrefined_mesh_path)  





block_connects = []
for cellblock in cells:
    block_connects.append(cellblock[1])


new_block_connects, new_cell_data, element_map = make_new_block_from_elementids(block_connects,
                                                                                ele_per_existing_block_for_new_block,
                                                                                None)





# correct side_sets using element_map, as element indices change due to new block
side_sets_old = copy.deepcopy(side_sets)
for key, c_set in side_sets.items():
    new_indices = [element_map.get(elem_id, elem_id) for elem_id in c_set[1]]
    side_sets[key] = tuple([side_sets[key][0],np.array(new_indices)])


# construct new mesh data with subdomain-box block
cells = []
for connect in new_block_connects:
    meshio_type = meshio_type_from_dim[np.shape(connect)[1]]
    cells.append((meshio_type, connect))

exio.write(os.path.join(script_dir,"440k_KlinikumBogenhausen_mesh_ref_block.e"), #"/Users/floriankonrad/Dropbox/5_numys/5_Numys_Projects/03_Kilinkum_Bogenhausen/02_Meshing/00_exports/440k_KlinikumBogenhausen_mesh_ref_block.e",
           points,cells,
      side_sets = side_sets,
      point_sets = point_sets,
      point_data = point_data,
      cell_data=cell_data,
      time_vals = time_vals,
      eb_names = eb_names)











#%% testing 



import netCDF4 
#testing
nc = netCDF4.Dataset(unrefined_mesh_path)
len(nc["ss_names"][:].data)
print(nc.variables.keys())
sides_ex = nc["side_ss1"][:].data
elem_ss_ex = nc["elem_ss1"][:].data

ns_status = nc["ns_status"][:].data
ns_prop1 = nc["ns_prop1"][:].data

ss_status = nc["ss_status"][:].data
ss_prop1 = nc["ss_prop1"][:].data

nc["elem_num_map"][:].data
nc["elem_ss1"][:].data

with open("exodus_structure_og.txt", "w") as file:
    print(nc, file=file)
    print(nc.dimensions, file=file)
    print(nc.variables, file=file)



var_keys = list(nc.variables.keys())


nc.close()




