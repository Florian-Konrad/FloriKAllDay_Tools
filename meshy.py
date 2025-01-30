# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:28:18 2023

@author: KONRAD.FLORIAN
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import pyvista as pv
import math
import statistics
from collections import Counter

# Get the current script directory
current_script_directory = os.path.dirname(__file__)
# Get the parent directory of the script
parent_directory = os.path.dirname(current_script_directory)

import sys
sys.path.append(parent_directory) 
import lib.meshio_ex as meshio_ex
import numpy as np
from lib import interswag as isw



meshio_type_from_dim = {2: "line",
                        3: "triangle",
                        4: "tetra",
                        8: "hexahedron"}




def get_node_id_from_coords(mesh_node_coords,searchnodecoords):
    '''
    search for nodes in a mesh using their coordinates
    and return their node id

    Parameters
    ----------
    mesh_node_coords : array (,3) of floats
        x,y,z coordinates of all nodes of a mesh.
    searchnodecoords : array (,3) of floats
        nodes to search for, must contain x,y,z coordinates of nodes existing in mesh_node_coords.

    Returns
    -------
    node_ids : array (,1) of ints
        node ids of the input coords.

    '''
    l = len(searchnodecoords)
    isw.print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    node_ids = []
    for i,node in enumerate(searchnodecoords):
        is_close = np.isclose(mesh_node_coords, node,rtol=1e-6)
        matching_rows = np.where(is_close.all(axis=1))
        # check that only one match should be found
        if len(matching_rows[0]) < 1:
            print("WARNING: No matching node found for coordinates: {}".format(node))
            print("Check if mesh_node_coords and searchnodecoords belong together and have been generated from the same meshfile")
        elif len(matching_rows[0]) > 1:
            print("WARNING: More than one node found for coordinates: {}".format(node))
        else:
            node_ids.append(matching_rows[0][0])
        #update progressbar:
        isw.print_progress(i + 1, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
        
    return node_ids




def move_nodes_meshio_mesh(meshfile,existing_nodes,corresponding_new_coord):
    '''
    
    simple function to change coordinates of existing nodes
    does not prevent collisions or other problems!!
    new node locations must make sense
    
    Parameters
    ----------
    meshfile : string
        path and name to meshfile, will be loaded with meshio.
    existing_nodes : array of floats
        contains x,y,z coordinates of nodes existing in the meshfile that should be moved.
    corresponding_new_coord : array of floats
        must have same length as existing_nodes
        conatins new x,y,z coordinates.

    Returns
    -------
    meshio mesh object, 
    array of mesh node coords,
    array of node indices of manipulated mesh nodes

    '''
    mesh = meshio_ex.read(meshfile)
    mesh_node_coords = mesh.points
    
    manip_ids = get_node_id_from_coords(mesh_node_coords,existing_nodes)
    
    for i, manip_id in enumerate(manip_ids):     
        new_node_coords = corresponding_new_coord[i]
        mesh_node_coords[manip_id] = new_node_coords
    
    return mesh, mesh_node_coords, manip_ids


def convert_cell_data(cell_data,cell_data_names,cell_data_convertion_factors):
    '''
    use to convert units of cell data by multiplication with a conversion factor

    Parameters
    ----------
    cell_data : dict
        must be a dictionary, eg meshio cell_data dict containing a cell_data key/name
        with a number of arrays of cell data values. the number of arrays depends on the block number in the mesh.
    cell_data_names : list of strings
        a list of all cell_data keys to which the values should be converted.
    cell_data_convertion_factors : list of floats
        must be of same length as cell_data_names, contains conversion factors for each cell_data array set.

    Returns
    -------
    cell_data : TYPE
        converte cell data dict.

    '''
    
    for i, name in enumerate(cell_data_names):
        # get current conversion factor
        conv_fact = cell_data_convertion_factors[i]
        # apply conversion on each block cell data
        # number of block is derived through np.shape(cell_data[name])[0]
        for ic in range(np.shape(cell_data[name][0])[0]):
            cell_data[name][0][ic] = cell_data[name][0][ic] * conv_fact
    
    return cell_data

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


def search_element_ids_around_nodes(block_connects,
                                    node_ids):
    '''
    Searching for elements that contain the specified nodes

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
            
    node_ids : list of ints
        List of node ids around which the elements will be searched for.

    Returns
    -------
    ele_around_node_ids_per_block : list of lists
        contains the desired list of cell ids for each block existing in the block_connects.

    '''
    # find all element indices that contain the well nodes
    ele_around_node_ids_per_block = []
    # search for wellnodes in elements of all blocks of multiblock mesh
    for cellblock in block_connects:
        ele_ids_cur_block_cont_n = np.array([])
        for node in node_ids:
            ix_ele_cont_node = np.where(np.isin(cellblock, node))[0]
            ele_ids_cur_block_cont_n = np.r_[ele_ids_cur_block_cont_n,ix_ele_cont_node].astype(int)
        # most elements will be found double, therefore duplictaes need to be removed
        ele_around_node_ids_per_block.append(np.unique(ele_ids_cur_block_cont_n))
        
    return ele_around_node_ids_per_block



def find_points_in_box(points, bottom_left, top_right):

    '''
    Parameters
    ----------

    Returns
    -------
    '''

    points_in_box = np.empty((0, 3))
    point_IDs_in_box = np.array([])

    for ID, point in enumerate(points):
        if bottom_left[0] <= point[0] <= top_right[0] and bottom_left[1] <= point[1] <= top_right[1] and bottom_left[2] <= point[2] <= top_right[2]:
            points_in_box = np.r_[points_in_box,[point]]
            point_IDs_in_box = np.r_[point_IDs_in_box,ID].astype(int)

    return points_in_box,point_IDs_in_box


def calc_element_centroids(mesh):
    centroids_all_blocks = []
    for blk_id, block in enumerate(mesh.cells):
        centroids = np.zeros((len(block.data), 3))
        for i, cell in enumerate(block.data):
            # Convert 'cell' to integer indices
            cell = cell.astype(int)
            centroids[i] = np.mean(mesh.points[cell, :], axis=0)
        centroids_all_blocks.append(centroids)
    return centroids_all_blocks



def subdomainblock_from_bounding_box(mesh_obj, bottom_left, top_right):
    '''
    uses a bounding box defined by its lower left and upper right corner point
    to generate a new block in the input mesh

    Parameters
    ----------
    mesh_obj : meshio mesh object
        eg mesh = meshio_ex.read(input_mesh_file).
    bottom_left : list/array (3,)
        coordinates of bottom left corner point of boudning box.
    top_right : list/array (3,)
        coordinates of upper right corner point of boudning box..

    Returns
    -------
    new_mesh : meshio mesh object
        new containing additional block.

    '''
    
    mesh_nodes = mesh_obj.points 
    centroids_all_blocks = calc_element_centroids(mesh_obj)
    
    element_ids_in_box = []
    for current_block_centroids in centroids_all_blocks:
        centroids_in_box_c_blk, element_IDs_in_box_c_blk = find_points_in_box(current_block_centroids, bottom_left, top_right)
        element_ids_in_box.append(element_IDs_in_box_c_blk)

    block_connects = []
    for cellblock in mesh_obj.cells:
        block_connects.append(cellblock.data)  
    new_block_connects, new_cell_data = make_new_block_from_elementids(block_connects,
                                                                       element_ids_in_box,
                                                                       mesh_obj.cell_data)
    
    
    cells = []
    for connect in new_block_connects:
        meshio_type = meshio_type_from_dim[np.shape(connect)[1]]
        cells.append((meshio_type, connect))
    
    # make new meshio object with new data
    new_mesh = meshio_ex.Mesh(mesh_nodes,
                              cells,
                              point_sets=mesh_obj.point_sets,
                              cell_data=new_cell_data)
    
    return new_mesh


def add_nodeset_from_nodes_file(mesh_nodes,point_sets,nodes_file_name):
    '''
    uses a text file containing coordinates for selected mesh nodes searches
    for them in an array of the entire mesh nodes existing in a mesh
    and adds them to a point_set dictionary

    Parameters
    ----------
    mesh_nodes : array of floats (,3)
        array of the entire mesh nodes existing in a mesh
        is generated with meshio by using mesh.points on the meshio mesh object.
    point_sets : dict
        a meshio style point_set dictionary
        containing names and corresponding array of nodeids.
    nodes_file_name : string
        a filename / path that is read in using numpy
        containing only x,y,z values comma separated without header.

    Returns
    -------
    point_sets : dict
        a meshio style point_set dictionary
        containing the content of point_sets but has the desired additional point set added
        the name of the new point set is determined by using the file basename.

    '''
    
    base_name = nodes_file_name.split(".")[0]
    nodes = np.genfromtxt(nodes_file_name, delimiter=',')
    node_ids = get_node_id_from_coords(mesh_nodes,nodes)
    point_sets[base_name] = np.array(node_ids).astype(int)
    
    return point_sets

def find_nodes_with_element_gap(uniq_xy, nearest_point_list):

    '''
    Parameters
    ----------
    uniq_xy : Dataframe
        contains all possible combination of x&y in mesh
        size (,2)
    nearest_point_list : numpy array
        contains the nearest points to the well intersection on the mesh
        size (,3)


    Returns
    -------
    nearest_points_with_element_gap : numpy array
        successive points from nearest_point_list which the distance is large enough, 
        that gaps exist between the corresponding element when using search_element_ids_around_nodes 
        size (,3)
    '''

    #calculate difference between rows
    uniq_xy_diff_cell = uniq_xy.diff(axis=0)
    #drop nan
    uniq_xy_diff_cell.dropna(inplace=True)
    #drop zeros
    uniq_xy_diff_cell = uniq_xy_diff_cell[(uniq_xy_diff_cell[['X','Y']] != 0).all(axis=1)]
    #find the min distance of delta x & y for the dimension of smallest unit cell 
    uniq_xy_diff_cell = uniq_xy_diff_cell.abs().min()
    
    x_grid_size = uniq_xy_diff_cell['X']
    y_grid_size = uniq_xy_diff_cell['Y']
    
    # when the distance between two successive points is over the threshold distance,
    # the coordinates of such point should be saved
    threshold = ((2*x_grid_size)**2 + (2*y_grid_size)**2)**0.5
    #print('threshold distance is: {}'.format(threshold))
    
    nearest_points_with_element_gap = []
    
    for i in range(len(nearest_point_list)-1):
        
        point_before = nearest_point_list[i]
        point_after = nearest_point_list[i + 1]
    
        #calculate 2d distance
        distance = np.linalg.norm(point_after[:2] - point_before[:2])
    
        if distance >= threshold:
            
            nearest_points_with_element_gap.append(point_before)
            nearest_points_with_element_gap.append(point_after)
    
    #drop duplicated coordinates
    nearest_points_with_element_gap = [list(x) for x in set(tuple(x) for x in nearest_points_with_element_gap)]
    nearest_points_with_element_gap = np.array(nearest_points_with_element_gap).reshape(-1, 3)
    # sort z in descending order
    nearest_points_with_element_gap = nearest_points_with_element_gap[np.argsort(-nearest_points_with_element_gap[:, 2])]

    return nearest_points_with_element_gap

def get_bottom_left_top_right(nearest_points_with_element_gap):


    '''

    Parameters
    ----------
    nearest_points_with_element_gap : numpy array
        successive points from nearest_point_list which the distance is large enough, 
        that gaps exist between the corresponding element when using search_element_ids_around_nodes 
        size (,3)


    Returns
    -------
    bl_tr : numpy array
        all coordinates of bottom_left and top_right
        first three columns: bottom_left coordinates, the last three columns: top_right coordinates 
        size (,6)
    
    '''
    # tr for top right
    tr_x = []
    tr_y = []
    tr_z = []
    # bl for bottom left
    bl_x = []
    bl_y = []
    bl_z = []
    
    for i in range(len(nearest_points_with_element_gap)-1):
        
        # point_before and point_after represent the two successive points
        point_before = nearest_points_with_element_gap[i]
        point_after = nearest_points_with_element_gap[i + 1]
        
        # four cases of how the direction of well path could be oriented
    
        #  left upwards/ leftwards
        if point_before[0] > point_after[0] and point_before[1] <= point_after[1] and point_before[2] > point_after[2]:
            
            tr_x.append(point_before[0])
            tr_y.append(point_after[1])
            tr_z.append(point_before[2]) 
            
            bl_x.append(point_after[0]) 
            bl_y.append(point_before[1]) 
            bl_z.append(point_after[2]) 
                  
        # right upwards/ upwards
        elif point_before[0] <= point_after[0] and point_before[1] < point_after[1] and point_before[2] > point_after[2]:
            
            tr_x.append(point_after[0])
            tr_y.append(point_after[1])
            tr_z.append(point_before[2]) 
            
            bl_x.append(point_before[0]) 
            bl_y.append(point_before[1]) 
            bl_z.append(point_after[2]) 

        # left downwards/  downwards
        elif point_before[0] >= point_after[0] and point_before[1] > point_after[1] and point_before[2] > point_after[2]:
            
            tr_x.append(point_before[0])
            tr_y.append(point_before[1])
            tr_z.append(point_before[2]) 
            
            bl_x.append(point_after[0]) 
            bl_y.append(point_after[1]) 
            bl_z.append(point_after[2]) 

        # right downwards/ rightwards 
        elif point_before[0] < point_after[0] and point_before[1] >= point_after[1] and point_before[2] > point_after[2]:
            
            tr_x.append(point_after[0])
            tr_y.append(point_before[1])
            tr_z.append(point_before[2]) 
            
            bl_x.append(point_before[0]) 
            bl_y.append(point_after[1]) 
            bl_z.append(point_after[2]) 
        
        tr = list(zip(tr_x, tr_y, tr_z))
        bl = list(zip(bl_x, bl_y, bl_z))
        
        tr = np.array(tr)
        bl = np.array(bl)

        # merge the arrays of bottom_left and top_right together
        #first three columns: bottom_left coordinates, the last three columns: top_right coordinates 
        bl_tr = np.c_[bl, tr]

    return bl_tr

def find_missing_element_ids_around_well(mesh, nearest_points, uniq_xy):

    '''
    Parameters
    ----------

    uniq_xy : Dataframe
        contains all possible combination of x&y in mesh
        size (,2)
    nearest_point_list : numpy array
        contains the nearest points to the well intersection on the mesh
        size (,3)


    Returns
    -------
    missing_element_ids_around_well : list (of lists)
        contains the missing element ids between two successive nodes with element gaps

    
    '''
    # find the points from nearest_point_lists that require extra sub-boxes in between
    nearest_points_with_element_gap = find_nodes_with_element_gap(uniq_xy, nearest_points)
    print(nearest_points_with_element_gap)

    #if the array is not empty
    if len(nearest_points_with_element_gap) > 0: 
        # create virtual points of nearest_points_with_element_gap, 
        # so that the sub boxes follow the rule of top_right & bottom_left
        bottom_left_top_right = get_bottom_left_top_right(nearest_points_with_element_gap)

        centroids_all_blocks = calc_element_centroids(mesh)
        
        missing_element_ids_around_well = []
        #loop through all available blocks in mesh   
        for current_block_centroids in centroids_all_blocks:
            element_ids_in_box_c_blk_combined = []  # New list for each iteration
            
            for i in range(len(bottom_left_top_right)):
                
                centroids_in_box_c_blk, element_ids_in_box_c_blk = find_points_in_box(current_block_centroids, bottom_left_top_right[:,0:3][i], bottom_left_top_right[:,3:][i])
                element_ids_in_box_c_blk_combined.extend(element_ids_in_box_c_blk)

            missing_element_ids_around_well.append(element_ids_in_box_c_blk_combined)

    else:
        # if the list nearest_points_with_element_gap is empty
        # that means all box of successive points are well connected
        # no missing_element_ids_around_well should be added
        missing_element_ids_around_well = []
        print('No missing elements exist!!')


    return missing_element_ids_around_well


def get_discretization_on_surface(surface_points):
    
    # initiate progressbar, can come from interswag(isw) lib or reseng --> misc
    l = len(surface_points)
    isw.print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    
    # iterate over slice, select a point, remove point from slice and build KDtree for filtered slice and calculate distance
    distances = []
    for i, point in enumerate(surface_points):
        arr = np.delete(surface_points, i, axis=0)
        kdtree_arr = KDTree(arr)
        distanz, _ = kdtree_arr.query(point)
        distances.append(distanz)
        isw.print_progress(i + 1, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    
    surface_discr_extreme = [max(distances),min(distances)]

    return surface_discr_extreme

def get_discretization_from_nodes(mesh_points):

    # make a df so its easier to get subset
    df = pd.DataFrame(mesh_points)
    # get a slice of mesh nodes made out of unique x,y combinations
    # does not necessary correspond to a mesh layer but doesnt matter in this context
    xy_slice = df.drop_duplicates(subset=[0,1], ignore_index=True).to_numpy()
    
    # initiate progressbar, can come from interswag(isw) lib or reseng --> misc
    l = len(xy_slice)
    isw.print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    
    # iterate over slice, select a point, remove point from slice and build KDtree for filtered slice and calculate distance
    distances = []
    for i, point in enumerate(xy_slice):
        arr = np.delete(xy_slice, i, axis=0)
        kdtree_arr = KDTree(arr)
        distanz, _ = kdtree_arr.query(point)
        distances.append(distanz)
        isw.print_progress(i + 1, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    
    # take one unique combination of x & y coords and look for all possible z-values = Mesh edge
    # here all information about vertical discretization is stored
    # the exact values correspond of course only to this one egde
    # but to get a magnitude for the verttical discretization this is probably enough
    some_mesh_edge =df.loc[(df[0] == xy_slice[0][0]) & (df[1] == xy_slice[0][1])]
    edge_df = pd.DataFrame(some_mesh_edge)
    edge_df = edge_df.sort_values(by=2)
    z_diff = edge_df.diff().abs().dropna() 
    z_diff = z_diff.loc[z_diff[2]>0][2]
    
    horizontal_discr_extreme = [max(z_diff),min(z_diff)]
    vertical_discr_extreme = [max(distances),min(distances)]

    return horizontal_discr_extreme,vertical_discr_extreme

def magnitude (value):
    if (value == 0): return 0
    return int(math.floor(math.log10(abs(value))))


def get_mesh_nodes_on_fault_surface(mesh_points,fault_point_cloud):
    
    # input type check
    if isinstance(fault_point_cloud, np.ndarray):
        fault_point_cloud = fault_point_cloud
    elif isinstance(fault_point_cloud, pd.DataFrame):
        fault_point_cloud = fault_point_cloud.to_numpy()
    elif isinstance(fault_point_cloud, list):
        fault_point_cloud = np.array(fault_point_cloud) 
    else:
        print("Input Type of fault point cloud is unknown, please input either, DataFrame, Array or List!")
    
    print()
    print("Selecting mesh nodes only with ~1000m distance to fault surface!")
    fault_mesh = pv.PolyData(fault_point_cloud).delaunay_2d()
    kdtree_coarse = KDTree(fault_mesh.points)
    distanzen_coarse, _ = kdtree_coarse.query(mesh_points)
    df = pd.DataFrame(np.c_[mesh_points,distanzen_coarse],columns=["x","y","z","dist_to_fault_coarse"])
    mesh_points_coars_selected = df.loc[df["dist_to_fault_coarse"] < 1000].to_numpy()[:,:3]
    
    
    print()
    print("Getting discretization values for fault points...")
    surface_discr_extreme = get_discretization_on_surface(fault_point_cloud)
    print()
    print("Getting discretization values for mesh points around fault zone...")
    mesh_discr = get_discretization_from_nodes(mesh_points_coars_selected)
    
    fault_min_discr = min(surface_discr_extreme)
    mesh_min_discr = min(min(mesh_discr))
    
    if magnitude(fault_min_discr) > magnitude(mesh_min_discr):
        fault_subdivide_factor = 3 + magnitude(fault_min_discr) - magnitude(mesh_min_discr)
    else:
        fault_subdivide_factor = 2
    
    print()
    print("Looking for nodes on fault surface. Might take a while as pointcloud delaunay has to be very fine for this algo to work")
    
    
    fault_mesh_ref = fault_mesh.subdivide(fault_subdivide_factor, subfilter='linear')
    fault_mesh_ref.save('fault_mesh_ref.vtk')
    kdtree = KDTree(fault_mesh_ref.points)
    distanzen, _ = kdtree.query(mesh_points_coars_selected)
    
    df = pd.DataFrame(np.c_[mesh_points_coars_selected,distanzen],columns=["x","y","z","dist_to_fault"])
    nodes_on_fault = df.loc[df["dist_to_fault"] < mesh_min_discr].to_numpy()
    print("{} nodes in the mesh have been found".format(len(nodes_on_fault)))
    
    return nodes_on_fault


def extract_fault_edges_nodes(uniq_xy, z_values_at_unique_xy_df):
    
    
    
    ##### Step 00 find the length of z values with the most occurence #####
    
    # Extract Z values from the DataFrame
    z_values_lists = [z_values_at_unique_xy_df.iloc[i].dropna().tolist() for i in range(len(z_values_at_unique_xy_df))]
    
    # Calculate the length of each inner list
    lengths_of_lists = [len(inner_list) for inner_list in z_values_lists]
    
    # Find the most common length and its count
    counter = Counter(lengths_of_lists)
    most_common_length, occurrences = counter.most_common(1)[0]
    
    
    ##### step 01 calculate stz pointcloud #####
    
    indices_above_most_common_length = [i for i, length in enumerate(lengths_of_lists) if length > most_common_length]
    
    # Filter rows in uniq_xy with a length above the most_common_length of Z values
    filtered_uniq_xy = uniq_xy.iloc[indices_above_most_common_length]
    
    # Extract corresponding Z values from z_values_at_unique_xy
    z_values_above_most_common_length = z_values_at_unique_xy_df.iloc[indices_above_most_common_length]
    
    # Combine all rows in filtered_uniq_xy with a length above the most_common_length of Z values to form a single point cloud
    combined_point_cloud_above_most_common_length = []
    for index, corresponding_xy_row in filtered_uniq_xy.iterrows():
        for z in z_values_above_most_common_length.loc[index].dropna():
            combined_point_cloud_above_most_common_length.append((corresponding_xy_row["X"], corresponding_xy_row["Y"], z))
    
    # Create a DataFrame from the combined_point_cloud_above_most_common_length
    nodes_on_all_faults = pd.DataFrame(combined_point_cloud_above_most_common_length, columns=["X", "Y", "Z"])
    
    
    ##### step 02 calculate pointcloud without fault traces #####
    
    indices_most_common_length = [i for i, length in enumerate(lengths_of_lists) if length <= most_common_length]
    
    # Filter rows in uniq_xy with a length above the most_common_length of Z values
    filtered_uniq_xy = uniq_xy.iloc[indices_most_common_length]
    
    # Extract corresponding Z values from z_values_at_unique_xy
    z_values_above_most_common_length = z_values_at_unique_xy_df.iloc[indices_most_common_length]
    
    mesh_layer_coords_no_fault_traces = pd.concat([filtered_uniq_xy,z_values_above_most_common_length], axis=1)
    
    mesh_layer_coords_no_fault_traces.dropna(axis=1, inplace=True)
    mesh_layer_coords_no_fault_traces = mesh_layer_coords_no_fault_traces.reset_index(drop=True)
    
    
    return mesh_layer_coords_no_fault_traces, nodes_on_all_faults




