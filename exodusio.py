#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:49:55 2025

Reading and writing of exodus meshes 
with focus on moose files
to enable manipulation of mesh properties using python

@author: floriankonrad
"""


import numpy as np
import re
import netCDF4
import datetime


exodus_to_meshio_type = {
    "SPHERE": "vertex",
    # curves
    "BEAM": "line",
    "BEAM2": "line",
    "BEAM3": "line3",
    "BAR2": "line",
    "EDGE2": "line",
    "TRUSS": "line",
    # surfaces
    "SHELL": "quad",
    "SHELL4": "quad",
    "SHELL8": "quad8",
    "SHELL9": "quad9",
    "QUAD": "quad",
    "QUAD4": "quad",
    "QUAD5": "quad5",
    "QUAD8": "quad8",
    "QUAD9": "quad9",
    #
    "TRI": "triangle",
    "TRIANGLE": "triangle",
    "TRI3": "triangle",
    "TRI6": "triangle6",
    "TRI7": "triangle7",
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL6': 'triangle6',
    # 'TRISHELL7': 'triangle',
    #
    # volumes
    "HEX": "hexahedron",
    "HEXAHEDRON": "hexahedron",
    "HEX8": "hexahedron",
    "HEX9": "hexahedron9",
    "HEX20": "hexahedron20",
    "HEX27": "hexahedron27",
    #
    "TETRA": "tetra",
    "TETRA4": "tetra4",
    "TET4": "tetra4",
    "TETRA8": "tetra8",
    "TETRA10": "tetra10",
    "TETRA14": "tetra14",
    #
    "PYRAMID": "pyramid",
    "WEDGE": "wedge",
}
meshio_to_exodus_type = {v: k for k, v in exodus_to_meshio_type.items()}



def categorize(names):
    # Check if there are any <name>R, <name>Z tuples or <name>X, <name>Y, <name>Z
    # triplets in the point data. If yes, they belong together.
    single = []
    double = []
    triple = []
    is_accounted_for = [False] * len(names)
    k = 0
    while True:
        if k == len(names):
            break
        if is_accounted_for[k]:
            k += 1
            continue
        name = names[k]
        if name[-1] == "X":
            ix = k
            try:
                iy = names.index(name[:-1] + "Y")
            except ValueError:
                iy = None
            try:
                iz = names.index(name[:-1] + "Z")
            except ValueError:
                iz = None
            if iy and iz:
                triple.append((name[:-1], ix, iy, iz))
                is_accounted_for[ix] = True
                is_accounted_for[iy] = True
                is_accounted_for[iz] = True
            else:
                single.append((name, ix))
                is_accounted_for[ix] = True
        elif name[-2:] == "_R":
            ir = k
            try:
                iz = names.index(name[:-2] + "_Z")
            except ValueError:
                iz = None
            if iz:
                double.append((name[:-2], ir, iz))
                is_accounted_for[ir] = True
                is_accounted_for[iz] = True
            else:
                single.append((name, ir))
                is_accounted_for[ir] = True
        else:
            single.append((name, k))
            is_accounted_for[k] = True

        k += 1

    if not all(is_accounted_for):
        raise print("Read Error")
    return single, double, triple



def read(filename):
    '''
    Read an exodus file and return data.
    The data can be manipulated and used together with write()
    to store into a new exodus file

    Parameters
    ----------
    filename : str
        Path and filename to store new exodus mesh.

    Returns
    -------
    points : TYPE
        containing coords of mesh nodes, usually shape (n,3) for 3D meshes.
    cells : list of tuples of str and array
        each tuple is decribing the elements of a mesh block,
        each tuple consits of a element type string using meshio nomenclature (eg. "triangle", "tetra4", "line")
        and an array containing the element definitions using node indices 0-based, eg an array defining
        tetrahedral elements has a shape of n,4 with n = number of elements in that block.
    point_data : dict of arrays
        defining nodal variable values, which can be optionally temporally variable,
        dict keys are nodal variables names,
        dict values are arrays of shape t,n with t = number of timesteps and n = number of nodes for temporal variables
        or of shape n for non-temporal variables, time_vals must be empty then!. The default is {}.
    cell_data : dict of arrays of float or of arrays of floats
        defining cell variable values, which can be optionally temporally variable,
        dict keys are cell variables names,
        dict values are a list of arrays of shape t,n, arrays correpsond to each block,
        with t = number of timesteps and n = number of cells in current block for temporal variables
        or list of shape n for non-temporal variables, time_vals must be empty then!. The default is {}.
    point_sets : dict of arrays
        defining point sets aka node sets,
        dict keys are side/node set names,
        dict values are arrays defining nodes by their index 0-based. The default is {}.
    info : str
        exodus mesh info.
    time_vals : array/list of float
        array of times of all timesteps,
        for non temporal data leave empty so default value is set. The default is [0.0].
    eb_names : list of str
        defining block names. The default is [].
    side_sets : dict of tuple
        defining sidesets used to apply boundary conditions,
        dict keys are side set names,
        tuples consist of element side definition from exodus(sides_ss) and element ID (elem_ss), both 1-based by exodus definition. The default is {}. 
    
    '''

    with netCDF4.Dataset(filename) as nc:
        points = np.zeros((len(nc.dimensions["num_nodes"]), 3))
        point_data_names = []
        cell_data_names = []
        pnt_dta = {}
        cll_dta = {}
        cells = []
        ns_names = []
        ss_names = []
        eb_names = []
        ns = []
        sides_ss = []
        elem_ss = []
        point_sets = {}
        side_sets = {}
        info = []
        time_info = nc.variables['time_whole'][:]
        if len(time_info) > 1:
            time_vals = time_info.filled()
        else:
            time_vals = 0.0
        


        

        for key, value in nc.variables.items():
            
            if key == "info_records":
                value.set_auto_mask(False)
                for c in value[:]:
                    try:
                        info += [b"".join(c).decode("UTF-8")]
                    except UnicodeDecodeError:
                        # https://github.com/nschloe/meshio/issues/983
                        pass
            elif key == "qa_records":
                value.set_auto_mask(False)
                for val in value:
                    info += [b"".join(c).decode("UTF-8") for c in val[:]]
            elif key[:7] == "connect":
                meshio_type = exodus_to_meshio_type[value.elem_type.upper()]
                cells.append((meshio_type, value[:].filled() - 1))
            elif key == "coord":
                points = nc.variables["coord"][:].T
            elif key == "coordx":
                points[:, 0] = value[:]
            elif key == "coordy":
                points[:, 1] = value[:]
            elif key == "coordz":
                points[:, 2] = value[:]
            elif key == "name_nod_var":
                value.set_auto_mask(False)
                point_data_names = [b"".join(c).decode("UTF-8") for c in value[:]]
            elif key[:12] == "vals_nod_var":
                idx = 0 if len(key) == 12 else int(key[12:]) - 1
                value.set_auto_mask(False)
                
                # FK Meshio change

                pnt_dta[idx] = value[:]

            elif key == "name_elem_var":
                value.set_auto_mask(False)
                cell_data_names = [b"".join(c).decode("UTF-8") for c in value[:]]
                cell_data_names = [name if len(name) > 0 else str(i) for i,name in enumerate(cell_data_names)]

                # check if a variable is defined on all blocks, if not define it for missing blocks with nan values
                # so the file can still be loaded
                totel_varnumber = len(nc.variables['name_elem_var'])
                total_blocknumber = len(nc.variables['eb_prop1'])
                variable_list = nc.variables.keys()

                var_on_block_count = []
                for var_id in range(totel_varnumber):
                    pattern = re.compile('vals_elem_var{}eb.*$'.format(var_id+1))
                    matching_entries = [string for string in variable_list if pattern.match(string)]
                    var_on_block_count.append(len(matching_entries))
                var_on_block_count = np.array(var_on_block_count)
                
                var_indices = np.where(var_on_block_count != total_blocknumber)
                for idx in var_indices[0]:
                    for block in range(total_blocknumber):
                        key = 'vals_elem_var{}eb{}'.format(idx + 1,block + 1)
                        if key not in variable_list:
                            idx = int(idx)
                            if idx not in cll_dta:
                                cll_dta[idx] = {}
                            cll_dta[idx][block] = np.array(len(nc.variables['connect{}'.format(block + 1)]) * [np.nan])
                
                
            elif key[:13] == "vals_elem_var":
                # eb: element block
                m = re.match("vals_elem_var(\\d+)?(?:eb(\\d+))?", key)
                idx = 0 if m.group(1) is None else int(m.group(1)) - 1
                block = 0 if m.group(2) is None else int(m.group(2)) - 1

                value.set_auto_mask(False)
                # For now only take the first value
                if idx not in cll_dta:
                    cll_dta[idx] = {}
                 
                # FK Meshio change
                cll_dta[idx][block] = value[:]


            elif key == "ns_names":
                value.set_auto_mask(False)
                ns_names = [b"".join(c).decode("UTF-8") for c in value[:]]
                ns_names = [name if len(name) > 0 else str(i) for i,name in enumerate(ns_names)]
            elif key == "eb_names":
                value.set_auto_mask(False)
                eb_names = [b"".join(c).decode("UTF-8") for c in value[:]]
                eb_names = [name if len(name) > 0 else str(i) for i,name in enumerate(eb_names)]
            elif key.startswith("node_ns"):  # Expected keys: node_ns1, node_ns2
                ns.append(value[:].filled() - 1)  # Exodus is 1-based
                
            elif key == "ss_names":
                value.set_auto_mask(False)
                ss_names = [b"".join(c).decode("UTF-8") for c in value[:]]
                ss_names = [name if len(name) > 0 else str(i) for i,name in enumerate(ss_names)]
                
            elif key.startswith("side_ss"): # Expected keys: side_ss1, side_ss2
                sides_ss.append(value[:].filled()) # Exodus is 1-based 
                
            elif key.startswith("elem_ss"): # Expected keys: elem_ss1, elem_ss2
                elem_ss.append(value[:].filled()) # Exodus is 1-based
                
        
        # making sure that dict order is correct as the fix for variables that arent defined on some block
        # adds empty values before main loop and messes up correct order
        cll_dta_new = {}
        for i, ki in enumerate(cll_dta.keys()):
            cll_dta_new[i] = cll_dta[i]
        cll_dta = cll_dta_new



        # Check if there are any <name>R, <name>Z tuples or <name>X, <name>Y, <name>Z
        # triplets in the point data. If yes, they belong together.
        single, double, triple = categorize(point_data_names)

        point_data = {}
        for name, idx in single:
            point_data[name] = np.stack(pnt_dta[idx])
        for name, idx0, idx1 in double:
            point_data[name] = np.column_stack([np.stack(pnt_dta[idx0]), np.stack(pnt_dta[idx1])])
        for name, idx0, idx1, idx2 in triple:
            point_data[name] = np.column_stack([np.stack(pnt_dta[idx0]), np.stack(pnt_dta[idx1]), np.stack(pnt_dta[idx2])])

        cell_data = {}
        
        for name, data in list(zip(cell_data_names, cll_dta.values())):
            cell_data[name] = list(data.values())

        # generate dicts for better function output
        point_sets = {name: dat for name, dat in zip(ns_names, ns)}
        side_sets = {name: (sides, elem) for name, sides, elem in zip(ss_names , sides_ss, elem_ss)}
        

    return points,cells,point_data,cell_data,point_sets,info,time_vals,eb_names,side_sets







def write(file_path,points,cells,
          side_sets = {},
          point_sets = {},
          point_data = {},
          cell_data = {},
          time_vals = [0.0],
          eb_names = []):
    '''
    Write a mesh to an exodus file.
    Required are only nodes and cell-definitions and a filename
    Optional input are point & side sets, point & cell data, temporal point & cell data and block names
    This is designed to work with files from MeshIT and Moose Framework
    in combination with paraview

    Parameters
    ----------
    file_path : str
        Path and filename to store new exodus mesh.
    points : array
        containing coords of mesh nodes, usually shape (n,3) for 3D meshes.
    cells : list of tuples of str and array
        each tuple is decribing the elements of a mesh block,
        each tuple consits of a element type string using meshio nomenclature (eg. "triangle", "tetra4", "line")
        and an array containing the element definitions using node indices 0-based, eg an array defining
        tetrahedral elements has a shape of n,4 with n = number of elements in that block.
    side_sets : dict of tuple, optional
        defining sidesets used to apply boundary conditions,
        dict keys are side set names,
        tuples consist of element side definition from exodus(sides_ss) and element ID (elem_ss), both 1-based by exodus definition. 
        The default is {}.
    point_sets : dict of arrays, optional
        defining point sets aka node sets,
        dict keys are side/node set names,
        dict values are arrays defining nodes by their index 0-based. The default is {}.
    point_data : dict of arrays, optional
        defining nodal variable values, which can be optionally temporally variable,
        dict keys are nodal variables names,
        dict values are arrays of shape t,n with t = number of timesteps and n = number of nodes for temporal variables
        or of shape n for non-temporal variables, time_vals must be empty then!. The default is {}.
    cell_data : dict of arrays of float or of arrays of floats, optional
        defining cell variable values, which can be optionally temporally variable,
        dict keys are cell variables names,
        dict values are a list of arrays of shape t,n, arrays correpsond to each block,
        with t = number of timesteps and n = number of cells in current block for temporal variables
        or list of shape n for non-temporal variables, time_vals must be empty then!. The default is {}.
    time_vals : array/list of float, optional
        array of times of all timesteps,
        for non temporal data leave empty so default value is set. The default is [0.0].
    eb_names : list of str, optional
        defining block names. The default is [].

    Returns
    -------
    None.

    '''


    with netCDF4.Dataset(file_path, "w") as rootgrp:
        now = datetime.datetime.now().isoformat()   
            
        # =============================================================================
        # =============================================================================
        # # Setting Dimensions
        # =============================================================================
        # =============================================================================
    
        # set global data
        rootgrp.api_version = np.float32(4.9) # kp woher die Werte kommen, laut sanadia docs:  Database version number – the version of the data objects stored in the file. This document describes database version is 4.72.
        rootgrp.version = np.float32(4.9) #np.float32(netCDF4.getlibversion().split()[0]) ?
        rootgrp.floating_point_word_size = 8
        rootgrp.file_size = 1
        rootgrp.maximum_name_length = 32
        rootgrp.int64_status = 0
        rootgrp.title = f"Created by florik, using netCDF4 in python, {now}"
        
        
        
        #########################
        # set required dimensions
        total_num_elems = sum(c[1].data.shape[0] for c in cells)
    
        rootgrp.createDimension("len_name", 256)
        rootgrp.createDimension("time_step", None)
        rootgrp.createDimension("num_dim", points.shape[1])
        rootgrp.createDimension("num_nodes", len(points))
        rootgrp.createDimension("num_elem", total_num_elems)
        rootgrp.createDimension("num_el_blk", len(cells))
        
        for k, cell_block in enumerate(cells):
            dim1 = f"num_el_in_blk{k + 1}"
            dim2 = f"num_nod_per_el{k + 1}"
            rootgrp.createDimension(dim1, cell_block[1].data.shape[0])
            rootgrp.createDimension(dim2, cell_block[1].data.shape[1])
        
        # not sure which global variables these are
        rootgrp.createDimension("num_glo_var", 4)
        #rootgrp.createDimension("num_info", 1607) number of information records ??? should be skippable contains moose simulation info
        rootgrp.createDimension("len_line", 81)
        
        
        
        #########################
        # set optional dimensions
        
        # sideset stuff
        if len(side_sets) > 0 :
            # set number of sidesets
            rootgrp.createDimension("num_side_sets", len(side_sets))
            # set length of each individual side set
            for k, side_set in enumerate(side_sets.values()):
                dim = f"num_side_ss{k+1}"
                rootgrp.createDimension(dim, len(side_set[0]))            
            
        # point set stuff
        if len(point_sets) > 0 :
            # set number of point sets
            rootgrp.createDimension("num_node_sets", len(point_sets))
            # set length of each individual point/node set
            for k, node_set in enumerate(point_sets.values()):
                dim = f"num_nod_ns{k+1}"
                rootgrp.createDimension(dim, len(node_set))
            
        # point data aka nodal variables
        if len(point_data) > 0:
            rootgrp.createDimension("num_nod_var", len(point_data))
        
        # cell data aka elemental variables  
        if len(cell_data) > 0:
            rootgrp.createDimension("num_elem_var", len(cell_data))
        
            

    
        # =============================================================================
        # =============================================================================
        # # Setting Variables
        # =============================================================================
        # =============================================================================
    

        #########################
        # set required variables
    
        # set time steps
        data = rootgrp.createVariable("time_whole", 'f8', dimensions='time_step')
        data[:] = time_vals
    
        # elemnt block global stuff
    
        # eb_status means probably 1 = activa 0 = inactive ? not confirmed
        data = rootgrp.createVariable("eb_status", 'i4', dimensions='num_el_blk')
        for k in range(len(cells)):
            data[k] = int(1) 
    
        # eb_prop1 means probably block ID
        data = rootgrp.createVariable("eb_prop1", "i4", dimensions="num_el_blk")
        for k in range(len(cells)):
            data[k] = k
    
        # node coords 
        coord_names = ["coordx","coordy","coordz"]
        for axis in range(points.shape[1]):
            name = coord_names[axis]
            data = rootgrp.createVariable(name,"f8",dimensions="num_nodes")
            data[:] = points[:,axis]
        
        # coordinate axis names
        coor_names = rootgrp.createVariable("coor_names", "S1", dimensions= ("num_dim", "len_name"))
    
        coor_names[0, 0] = b"X"
        coor_names[1, 0] = b"Y"
        if points.shape[1] == 3:
            coor_names[2, 0] = b"Z"
            
        # skipping node_map
    
        # cell block connectivities defining elements on each block
        for k, cell_block in enumerate(cells):
            dim1 = f"num_el_in_blk{k + 1}"
            dim2 = f"num_nod_per_el{k + 1}"
            data = rootgrp.createVariable(f"connect{k + 1}", 'i4', dimensions=(dim1, dim2))
            data.elem_type = meshio_to_exodus_type[cell_block[0]]
            # Exodus is 1-based
            data[:] = cell_block[1] + 1
    
        # skipping elem_num_map
            
        
        
        
        
        #########################
        # set optional variables
        
        
        # nodes set status, prop1, name, values
        if len(point_sets) > 0:
            # probably means active or inactive
            data = rootgrp.createVariable("ns_status", 'i4', dimensions='num_node_sets')
            for k in range(len(point_sets)):
                data[k] = int(1)
            
            # probably gives an ID
            data = rootgrp.createVariable("ns_prop1", 'i4', dimensions='num_node_sets')
            for k in range(len(point_sets)):
                data[k] = k
                
            # node set names
            data = rootgrp.createVariable("ns_names","S1",dimensions=("num_node_sets", "len_name"))
            for i, name in enumerate(point_sets): #point_sets is a dict
                char_list = [bytes(char, 'utf-8') for char in name]
                data[i,:len(char_list)] = char_list
                
            # nodesets (non temporal)
            for k, node_set in enumerate(point_sets.values()):
                ns_name = f"node_ns{k + 1}"
                dim = f"num_nod_ns{k + 1}"
                
                data = rootgrp.createVariable(ns_name, 'i4', dimensions=dim)
                # Exodus is 1-based
                data[:] = node_set + 1 # assuming 1-based input here
            
            
            
        
        
        # sideset status, prop1, name, values
        if len(side_sets) > 0:
            # probably means active or inactive
            data = rootgrp.createVariable("ss_status", 'i4', dimensions='num_side_sets')
            for k in range(len(side_sets)):
                data[k] = int(1)
                
            # probably gives an ID    
            data = rootgrp.createVariable("ss_prop1", 'i4', dimensions='num_side_sets')
            for k in range(len(side_sets)):
                data[k] = k
                
            # side set names 
            data = rootgrp.createVariable("ss_names","S1",dimensions=("num_side_sets", "len_name"))
            for i, name in enumerate(side_sets): #point_sets is a dict
                char_list = [bytes(char, 'utf-8') for char in name]
                data[i,:len(char_list)] = char_list
                
            # sidesets (non temporal)
            for k, (sides, elem) in enumerate(side_sets.values()):
                elem_name = f"elem_ss{k + 1}"
                sides_name = f"side_ss{k + 1}"
                dim = f"num_side_ss{k + 1}"
                
                data = rootgrp.createVariable(elem_name, 'i4', dimensions=dim)
                # Exodus is 1-based
                data[:] = elem # assuming 1-based input here, as I#M normally not generating sidesets myself but just reading it from an exodus file, its then 1-based already
                
                data = rootgrp.createVariable(sides_name, 'i4', dimensions=dim)
                # Exodus is 1-based
                data[:] = sides  # assuming 1-based input here, as I#M normally not generating sidesets myself but just reading it from an exodus file, its then 1-based already
                
        
            
        # element block names 
        if len(eb_names) > 0:
            data = rootgrp.createVariable("eb_names","S1",dimensions=("num_el_blk", "len_name"))
            for i, name in enumerate(eb_names):
                char_list = [bytes(char, 'utf-8') for char in name]
                data[i,:len(char_list)] = char_list
        
        
        
        
    
        # temporal node data
        if len(point_data) > 0 :
            # node variable names
            data = rootgrp.createVariable("name_nod_var", 'S1', dimensions=("num_nod_var", "len_name"))
            for i, n_data in enumerate(point_data):
                n_data
                char_list = [bytes(char, 'utf-8') for char in n_data]
                data[i,:len(char_list)] = char_list
                
            # node variable data   
            for i, n_data in enumerate(point_data):
                variable_name = f"vals_nod_var{i+1}"
                data = rootgrp.createVariable(variable_name, 'f8', dimensions=("time_step", "num_nodes"))
                data[:] = point_data[n_data]
            
    
        # temporal cell data
        if len(cell_data) > 0 :
            # cell variable names
            data = rootgrp.createVariable("name_elem_var", 'S1', dimensions=("num_elem_var", "len_name"))
            for i, c_data in enumerate(cell_data):
                c_data
                char_list = [bytes(char, 'utf-8') for char in c_data]
                data[i,:len(char_list)] = char_list
        
            # cell variable data 
            for i, c_data in enumerate(cell_data):
                for k, block_array in enumerate(cell_data[c_data]):
                    variable_name = f"vals_elem_var{i+1}eb{k+1}"
                    block_dim = f"num_el_in_blk{k+1}"
                    data = rootgrp.createVariable(variable_name, 'f8', dimensions=("time_step", block_dim))
                    data[:] = block_array
    
    
    return


#%% Example Usage

# import os

# file_path = "/Users/floriankonrad/Dropbox/5_numys/5_Numys_Projects/01_Stephanskirchen/1_simulation/2_SimulationFiles"
# exodusfile = "TH_Stephanskirchen_transient_v3_final_out.e"
# test_file = os.path.join(file_path,exodusfile)


# points,cells,point_data,cell_data,point_sets,info,time_vals,eb_names,side_sets = read(test_file)

# # calculate pressure change vs first timestep
# point_data["dp_python"] = point_data["pore_pressure"] - point_data["pore_pressure"][0]

# # calculate head_change from dp_python
# point_data["delta_head_python"] = point_data["dp_python"] / (999.793 * 9.8065 * 1e-5)

# # calculate head
# point_data["head_python"] = point_data["pore_pressure"] / (999.793 * 9.8065 * 1e-5)


# save_path = os.path.join(file_path,"new_file.e")


# write(save_path,points,cells,
#       side_sets = side_sets,
#       point_sets = point_sets,
#       point_data = point_data,
#       cell_data=cell_data,
#       time_vals = time_vals,
#       eb_names = eb_names)

















