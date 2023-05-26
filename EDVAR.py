# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 14:50:22 2023

@author: rstudio
"""


# ======================================================================== Libraries ==================================================================

# Opening the files
import os
import tifffile
from readlif.reader import LifFile
import read_lif

# Image processing
import numpy as np
import vedo
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.measure import label
from skimage.morphology import medial_axis
from scipy.ndimage import gaussian_filter

# Image analysis
import pandas as pd
import cv2
from skan import Skeleton, summarize
from statistics import mean 
from numpy.linalg import norm

# Visualisation
import pyvista as pv
import pyvistaqt as pq
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

# Statistics and plotting
from pathlib import Path # to get the filename only, without the extension
from scipy.stats import f_oneway, ttest_ind, levene, kstest, tukey_hsd, mannwhitneyu, kruskal
from scipy.stats import norm as normaldensityfunction
import seaborn as sns
import scikit_posthocs as sp

# Tkinter
import tkinter
from tkinter import messagebox
from tkinter import filedialog
import customtkinter







# ==================================================================== 3D Image Analysis ==========================================================================


"""
Read the lif 
"""

def extract_z_stack(self, file, nb_lif, time, channel_fluo):
    
    # First, the chosen file .lif, .tif or .tiff file is extracted. 
    # In a .lif, there are usually several series (corresponding to the series shown when opening the file on ImageJ).
    # The series' number and the channels are chosen by the user.
    # This file is a z-stack, with a fluo channel. There are thus a time, fluo_channel, z, y, x channels.
    
    file_extension = os.path.splitext(file)[1]
    
    # .lif
    if file_extension == '.lif':
        reader = read_lif.Reader(file)
        series = reader.getSeries() # series correspond to ImageJ's series
        chose = series[nb_lif]  # chooses image set n° "nb_lif" in the lif file
        image = chose.getFrame(T=time, channel=channel_fluo) # This is a z-stack. The shape of image is (z, y, x) : image[0] is for z=0.
        return image
    #.tif
    elif file_extension == '.tif' or '.tiff':
        image_z_t = tifffile.imread(file)
        if len(image_z_t.shape) == 3: # z, x, y ie no time
            image = image_z_t
        if len(image_z_t.shape) == 4: # t, z, x, y
            image = image_z_t[time]
        if len(image_z_t.shape) == 5: # t, z, c, x, y
            image = image_z_t[time, :, channel_fluo, :, :]
        return image
    
    # other extension, invalid
    else: 
        self.message = messagebox.showinfo(title = "Error", message = "The extension of the file should be '.tif', '.tiff' or '.lif'")
                


"""
Extract Metadata
"""

def metadata(file, nb_lif):
    
    # Second, we extract the metadata from the file.
    # The metadata gives us the exact dimensions for the microvessels.
    # It comes directly from the microscope file.
    
    file_extension = os.path.splitext(file)[1]
    
    if file_extension == '.lif':
        new = LifFile(file)
        img_list = [i for i in new.get_iter_image()]
        
        scale_x_px = img_list[nb_lif].info['scale_n'][1] # px/µm
        scale_y_px = img_list[nb_lif].info['scale_n'][2]
        scale_z_px = img_list[nb_lif].info['scale_n'][3]
        
        scale_x = 1/(img_list[nb_lif].info['scale_n'][1]) # µm/px
        scale_y = 1/(img_list[nb_lif].info['scale_n'][2])
        scale_z = 1/(img_list[nb_lif].info['scale_n'][3]) 
        
        x_total_um = scale_x*(img_list[nb_lif].info['dims_n'][1]) # µm, total
        y_total_um = scale_y*(img_list[nb_lif].info['dims_n'][2])
        depth = scale_z*(img_list[nb_lif].info['dims_n'][3])
        
        x_total = img_list[nb_lif].info['dims_n'][1] # pixels, total
        y_total = img_list[nb_lif].info['dims_n'][2]
        z_stacks = img_list[nb_lif].info['dims_n'][3] # nb of slices
        
    if file_extension == '.tif':
        
        with tifffile.TiffFile(file) as tif: 
            metadata_file = {}
            for tag in tif.pages[0].tags: # tag = metadata
                tag_name, tag_value = tag.name, tag.value
                metadata_file[tag_name] = str(tag_value) # add the tags in a dictionary

        image_description = metadata_file['ImageDescription'].split('\n') # a list inside metadata_file
        my_dict = {}
        for i in image_description:
            for j in i:
                my_dict[str(i.split('=')[0])] = str(i.split('=')[1])
        
        # scale_x_px
        x_scale_px_1 = metadata_file['XResolution'].replace("(", "") # from a string with '(a,b)' to numbers
        x_scale_px_2 = x_scale_px_1.replace(")", "")
        x_scale_px_3 = x_scale_px_2.split(',')
        a = float(x_scale_px_3[0])
        b = float(x_scale_px_3[1])
        scale_x_px = a/b 
        #scale_y_px
        y_scale_px_1 = metadata_file['YResolution'].replace("(", "")
        y_scale_px_2 = y_scale_px_1.replace(")", "")
        y_scale_px_3 = y_scale_px_2.split(',')
        c = float(y_scale_px_3[0])
        d = float(y_scale_px_3[1])
        scale_y_px = c/d
        
        # scale_x, scale_y, scale_z
        scale_x = 1/scale_x_px # µm/px
        scale_y = 1/scale_y_px # µm/px
        scale_z = float(my_dict['spacing']) # µm/px
        
        # scale_z_px
        scale_z_px = 1/scale_z

        # x_total, y_total, z_stacks
        x_total = int(metadata_file['ImageWidth']) # pixels
        y_total = int(metadata_file['ImageLength']) 
        z_stacks = float(my_dict['slices'])

        # x_total_um, y_total_um, depth
        x_total_um = scale_x*x_total
        y_total_um = scale_y*y_total
        depth = scale_z*z_stacks
        
    return(scale_x_px, scale_y_px, scale_z_px, scale_x, scale_y, scale_z, x_total, y_total, z_stacks, x_total_um, y_total_um, depth)



"""
Blur and threshold 
"""


def who_is_the_mask(image, segmentation, kernel):
    global mask
    
    # Third, we perform the image processing.
    # The user can either choose to create the mask via the platform,
    # - this option is fit for images not unevenly exposed -
    # or, one can input directly one's own masks.
    # This is the recommended option.
    # Either way, the user should always check the mesh and pop-up window of the skeleton,
    # to see if the results of the analysis are correct. 
    
    if segmentation == 'on': # input is confocal images
    
    # HISTOGRAM EQUALIZATION makes segmentation more robust.
        img_equal = exposure.equalize_adapthist(image, clip_limit=0.03)    
    
        im_blur = gaussian_filter(img_equal, sigma=kernel)
        th = threshold_otsu(im_blur)
        mask = np.array(im_blur < th, dtype=np.int8)
        mask_inv = np.array(im_blur > th, dtype=np.int8) # filled mask
        
        im_blur_skel = gaussian_filter(img_equal, sigma=3)
        th = threshold_otsu(im_blur_skel)
        mask_skel = np.array(im_blur_skel < th, dtype=np.int8)
        mask_skel_inv = np.array(im_blur_skel > th, dtype=np.int8)
        
    else:
        mask = image # input is the mask
        mask_inv = cv2.bitwise_not(image)
        
    return (mask, mask_inv, mask_skel, mask_skel_inv)




"""
Get the mesh
"""

def real_mesh(mask, mask_inv, scale_x, scale_y, scale_z):
    
    # From the mask, the mesh is computed.
    # This mesh will serve for morphological analysis.
    # Two meshes are the ouput of this function: 
    # a surface mesh (for the diameters), and a filled mesh (for the area and volume).
    
    mask_transposed = np.transpose(mask, axes = [2,1,0])
    vol = vedo.Volume(mask_transposed, spacing=(scale_x,scale_y,scale_z)) # µm/px
    surf=vol.isosurface()
    
    # Filled volume: inverted volume, thus vessels are filled.
    mask_inv_transposed = np.transpose(mask_inv, axes=[2,1,0])
    vol_inv_um = vedo.Volume(mask_inv_transposed, spacing=(scale_x,scale_y,scale_z)) # µm/px
    voxels = vol_inv_um.legosurface(vmin = 0.5).triangulate() # for the area: vtkMassProperties estimates the volume, the surface area, and the normalized shape index of a closed, triangle mesh.
    
    return(vol, surf, vol_inv_um, voxels)




"""
Skeleton in 3D
"""

def skeleton_3d_make(mask_skel_inv): 
    
    # Here, the 3D skeleton is computed.
    # The skeleton is performed on the gaussian-blurred mesh, to have accurate branches
    
    skeleton_3d = skeletonize(mask_skel_inv, method='lee') # px/µm
    return skeleton_3d




"""
Skeleton analysis
"""

def skeleton_3d_analysis(skeleton_3d, x_total, y_total, scale_x):    
    # The 3D skeleton is projected and re-skeletonized, to ensure the highest robustness as possible before the analysis.
    # Indeed, the 3D skeleton could be disconnected in some parts due to the segmentation.
    # The branches that were going through the z plane are projected on the (x,y) plane, thus their length is still taken into account, 
    # which gives more details than doing just a skeletonization on the z-average.

    # Analysis
    z,x,y = skeleton_3d.nonzero()
    sk_3d_projection = np.zeros((x_total, y_total)) # skeleton_3d projection
    sk_3d_projection[x, y] = 1
    sk_3d_projection_clean = skeletonize(sk_3d_projection)
    branch_data = summarize(Skeleton(sk_3d_projection_clean, spacing=scale_x))
    
    # coordinates junctions and endpoints
    branch0 = branch_data[branch_data['branch-type'] == 0]
    branch1 = branch_data[branch_data['branch-type'] == 1]
    endpoints = branch1[["image-coord-dst-0", "image-coord-dst-1"]].values.astype(np.int64)
    branch2 = branch_data[branch_data['branch-type'] == 2]
    junctions_transitory = branch2[["image-coord-dst-0", "image-coord-dst-1"]].values.astype(np.int64)
    junctions = list(set(map(tuple,junctions_transitory)))
    branch3 = branch_data[branch_data['branch-type'] == 3]

    # Parameters extraction
    # branches
    number_branches = len(branch_data['branch-type'])
    number_isolated_branches = len(branch0)
    number_endpoint_junctions_branches = len(branch1)
    number_junction_junction_branches = len(branch2)
    number_circular_branches = len(branch3)
    average_branch_length = mean(branch_data['branch-distance']) #µm
    total_length = sum(branch_data['branch-distance'])
    
    # junctions
    number_junctions = len(junctions)
    number_endpoints = len(endpoints)
    
    return sk_3d_projection_clean, branch_data, number_branches, number_isolated_branches, number_endpoint_junctions_branches, number_junction_junction_branches, number_circular_branches, number_junctions, number_endpoints, average_branch_length, total_length, endpoints, junctions


def summarize_props_skeleton(branch_data):
    coord_src_0 = branch_data['coord-src-0'] # natural space; x, contain the start point coordinates of each branch, for axes 0, 1, 2.
    coord_src_1 = branch_data['coord-src-1'] # y
    coord_dst_0 = branch_data['coord-dst-0'] # and end point coordinates of each branch, for axes 0, 1, 2.
    coord_dst_1 = branch_data['coord-dst-1']
    return coord_src_0, coord_src_1, coord_dst_0, coord_dst_1




"""
Parameters extraction: area, volume, diameters
"""

def area_volume_diameters(self, surf, voxels, image, mask_skel, scale_x, scale_y, scale_z, x_total_um, y_total_um, depth, coord_src_0, coord_dst_0, coord_src_1, coord_dst_1):
    

    # ================================   Area, Volume  ==================================================
    # We use the 'area' and 'volume' functions from Vedo.
    # These functions use VtkMassProperties.
    # The conditions of use of it are to have a closed, triangulated mesh.
    # We thus use the inverted, triangulated mesh: 'voxels'.
    
    area = voxels.triangulate().area() # µm² 
    volume = voxels.triangulate().volume() 
    # for checking purposes, use the following formula: volume_estimation = len(a)*scale_x*scale_y*scale_z, as volume = <nb of points> * <scale of a point>.
    volume_total_fibrin_gel = x_total_um*y_total_um*depth
    percentage_occupancy = volume/volume_total_fibrin_gel
    area_total_fibrin_gel = 2*x_total_um*y_total_um + 2*y_total_um*depth + 2*x_total_um*depth # area of the piece studied under the microscope
    
    

    # ================================   Diameter_horizontal  ==================================================
    # We study the regions outside of the vessels,
    # and hypothesize that the slice with the most 'holes' between branches is the 
    # common plane for getting access to the horizontal diameter.
    # We thus get the 'blobs' that are outside of the vessels for each slice,
    # by the calculus of the connected components on each slice.
    # We then take the slice where there is the highest nb of components to measure the horizontal diameters.
    
    num = 0
    i = 0
    for slices in mask_skel: # to label, we need the mask, the not-inverted one to have the holes between the vessels
        labels, num_labels = label(slices, return_num=True) 
        if num_labels > num:
            num = num_labels
            z_slice = i # get the slice of interest
        i+=1

    # We need to resize the slice of interest from pixels to µm
    # We then compute the medial axis on its blurred mask (for accuracy of the labelling), 
    # and then get the values of the distance map that are on the medial axis
    
    # Resize and mask
    slice_interest = cv2.resize(image[z_slice], dsize=(int(x_total_um), int(y_total_um)), interpolation=cv2.INTER_CUBIC) # resize the slice to have access to the right measures
    interest_blur = gaussian_filter(slice_interest, sigma=3)
    th = threshold_otsu(interest_blur)
    mask_interest_inv = np.array(interest_blur > th, dtype=int) # inverted mask for filled vessels
    # Medial axis and distance map
    sk_medial_axis, sk_distance_map = medial_axis(mask_interest_inv, return_distance=True)
    x,y = sk_medial_axis.nonzero() # we take the distance_map values only where the medial axis was computed
    diameter_horizontal = []
    for i in range(len(x)):
        diameter_horizontal.append(2*(sk_distance_map[x[i], y[i]])) # µm, as image resized above
    
    
    
    # =================================   Diameter_vertical   ==================================================
    # The assumption made here is that the vertical diameter is measured at the center of the vessel (where we have the biggest height of the vessel). 
    # The horizontal center of the vessels is where the skeleton was computed. 
    # To find the vertical diameter of each vessel, we choose to measure it at the center of each branch of the skeleton. 
    # We then compute the perpendicular of the vessel, at the center of each skeleton’s branch. 
    # This perpendicular intersects in two points with the mesh: first, at the top of the vessel, and second, at the bottom of the vessel. 
    # The diameter of the vessel at a specific point can thus be calculated as the distance between the two intersection points between the vessel edges. 
    # NB: we take the list of x,y centers of branches to compute it, 
    # in order to be sure to be in a real diameter (and not on the side, or on the border of a hole).
    
    diameter_vertical = []
    # For checking purposes
    ptss = []
    middle = []
    for i in range(len(coord_src_0)):
        middle_branch_point = [int((coord_src_0[i]+coord_dst_0[i])/2), int((coord_src_1[i]+coord_dst_1[i])/2)] # middle of the skeleton branch
        # To be noticed: the middles are not exactly at the middle of each branch, due to curved branches.
        pts = surf.intersect_with_line(p0 = [middle_branch_point[0], middle_branch_point[1], 0], p1=[middle_branch_point[0], middle_branch_point[1], int(depth)], tol=0) # the two intersection points
        
        if len(pts) == 2: # ensure not an infinite value from an opening of the surface mesh
            diameter_vertical.append(norm(pts[1] - pts[0])) # euclidean_distance
            
            # for checking purposes
            middle_branch_point.append(depth)
            middle.append(middle_branch_point)
            ptss.append(pts[1]) # only one point for display, checking purposes
    
    # =================================   Mean Diameters   ==================================================    
    
    if len(diameter_vertical) == 0:
        self.message = messagebox.showerror(title = "Error", 
                                            message = "Your data is not complete /n" +
                                            "The confocal images should be taken in the entire depth of the vessels /n" +
                                            "Here, your mesh is open at the top / at the bottom of the stack, /n" +
                                            "Due to the lack of images from the top / the bottom of the vessels. /n" +
                                            "Thus, the area, volume and vertical diameters either cannot be considered as accurate, /n" +
                                            "or cannot be computed. /n" +
                                            "This is indicated in the excel saved datasheet with N/A written for mean_diameter_vertical.")
        mean_diameter_horizontal = mean(diameter_horizontal)
        mean_diameter_vertical = "N/A"
    else:                                
        mean_diameter_horizontal = mean(diameter_horizontal)
        mean_diameter_vertical = mean(diameter_vertical)
    
    return area, volume, volume_total_fibrin_gel, area_total_fibrin_gel, percentage_occupancy, diameter_horizontal, diameter_vertical, mean_diameter_horizontal, mean_diameter_vertical




"""
Get the max projection
"""
def max_projection(image, x_total, y_total, x_total_um, y_total_um):
    max_proj = np.zeros((x_total,y_total))
    for i in image: # iterate on each z
        for position, pixel_value in enumerate (i):
            max_proj[position] += pixel_value
    max_proj_toplot = cv2.resize(max_proj, dsize=(int(x_total_um), int(y_total_um)), interpolation=cv2.INTER_CUBIC) 
    return max_proj, max_proj_toplot # max_proj to check on the skeleton in the pop-up window, and max_proj_to_plot in the display window











# ================================================================== GUEST USER INTERFACE ==================================================================

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        self.title("ƎDVAR")
        self.state('zoomed')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed
        
        
        
        
        
# ========================================================================= LAYOUT ==================================================================        


        # ================================================= create two frames: Left + Right =========================================================
        
        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # frame_left
        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")
        # frame_right
        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)
        
        

        # ================================================================== LEFT =====================================================================

        # configure grid layout (1x11), ie 1 column and 11 lines
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(8, weight = 9) # empty row as spacing
        self.frame_left.grid_rowconfigure((9,10,11,12), weight=1) 
        self.frame_left.grid_rowconfigure(16, weight = 9)
        self.frame_left.grid_rowconfigure(18, minsize=10)  # empty row with minsize as spacing
        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="ƎDVAR",
                                              text_font=("Roboto Medium", -16))
        self.label_1.grid(row=1, column=0, pady=10, padx=10)
        
        self.button_0 = customtkinter.CTkButton(master=self.frame_left,
                                                text="BATCH",
                                                fg_color= 'maroon1',
                                                command=self.batch)
        self.button_0.grid(row=1, column=0, pady=10, padx=20)
        
        
        # OPEN the .lif / .tif / .tiff file
        self.combobox_time = customtkinter.CTkComboBox(master=self.frame_left,
                                                    values=["Timepoint", "0", "1", "2", "3", "4", "5"])
        self.combobox_time.grid(row=2, column=0, pady=10, padx=20)

        self.combobox_channel = customtkinter.CTkComboBox(master=self.frame_left,
                                                    values=["Channel", "0", "1", "2", "3", "4", "5"])
        self.combobox_channel.grid(row=3, column=0, pady=10, padx=20)
        
        self.combobox_nb_lif = customtkinter.CTkComboBox(master=self.frame_left,
                                                    values=["Series", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70"])
        self.combobox_nb_lif.grid(row=4, column=0, pady=10, padx=20)
        
        self.combobox_segmentation = customtkinter.CTkComboBox(master=self.frame_left,
                                                    values=["Segmentation", "on", "off"])
        self.combobox_segmentation.grid(row=5, column=0, pady=10, padx=20)


        # CHOOSE the file and SAVE the results
        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose File",
                                                command=self.open_file)
        self.button_1.grid(row=6, column=0, pady=10, padx=20)
        

        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Save parameters",
                                                command=self.save_parameters)
        self.button_2.grid(row=7, column=0, pady=10, padx=20)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Save mesh CFD",
                                                command=self.save_mesh_CFD)
        self.button_3.grid(row=8, column=0, pady=10, padx=20)
        
        
        # PLOTTING: Choose set of colors
        self.color_label = customtkinter.CTkLabel(master=self.frame_left, text="Groups' colors")
        self.color_label.grid(row = 9, column = 0, pady=2, padx=20, sticky="we")
        
        self.combobox_color_group1_var = customtkinter.StringVar(value="Color1")
        self.combobox_color_group1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                 values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                 variable =  self.combobox_color_group1_var,
                                                                 fg_color = "gray12")
        self.combobox_color_group1.grid(row=9, column = 0, pady=2, padx=20, sticky="we")

        self.combobox_color_group2_var = customtkinter.StringVar(value="Color2")
        self.combobox_color_group2 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                 values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                 variable =  self.combobox_color_group2_var,
                                                                 fg_color = "gray12")
        self.combobox_color_group2.grid(row = 10, column = 0, pady=2, padx=20, sticky="we")
        
        self.combobox_color_group3_var = customtkinter.StringVar(value="Color3")
        self.combobox_color_group3 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                 values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                 variable =  self.combobox_color_group3_var,
                                                                 fg_color = "gray12")
        self.combobox_color_group3.grid(row = 11, column = 0, pady=2, padx=20, sticky="we")
        
        self.combobox_color_group4_var = customtkinter.StringVar(value="Color4")
        self.combobox_color_group4 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                 values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                 variable =  self.combobox_color_group4_var,
                                                                 fg_color = "gray12")
        self.combobox_color_group4.grid(row = 12, column = 0, pady=2, padx=20, sticky="we")
        
        self.combobox_color_group5_var = customtkinter.StringVar(value="Color5")
        self.combobox_color_group5 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                 values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                 variable =  self.combobox_color_group5_var,
                                                                 fg_color = "gray12")
        self.combobox_color_group5.grid(row = 13, column = 0, pady=2, padx=20, sticky="we")

        self.combobox_color_group6_var = customtkinter.StringVar(value="Color6")
        self.combobox_color_group6 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                 values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                 variable =  self.combobox_color_group6_var,
                                                                 fg_color = "gray12")
        self.combobox_color_group6.grid(row = 14, column = 0, pady=2, padx=20, sticky="we")
        
        self.plot_button = customtkinter.CTkButton(master=self.frame_left, text="Plot",
                                                   command = self.button_plot)
        self.plot_button.grid(row = 15, column = 0, padx=2, pady=20, sticky="we")

        # APPEARANCE MODE
        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.label_mode.grid(row=17, column=0, pady=0, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=18, column=0, pady=10, padx=20, sticky="w")
        
        
        
        # ================================================================== RIGHT ==================================================================
        
        # configure grid layout (6x13) ie 6 columns, 13 lines
        self.frame_right.rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), weight=1)
        self.frame_right.columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)


        # Subframe with DISPLAY
        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=7, rowspan=3, pady=20, padx=20, sticky="nsew")
        self.frame_info.rowconfigure(0, weight=1) # configure grid layout (1x1)
        self.frame_info.columnconfigure(0, weight=1)
        
        # Subframe with PARAMETERS' ANALYSIS
        self.frame_parameters = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_parameters.grid(row=5, column=0, columnspan=7, rowspan=4, pady=20, padx=20, sticky="nsew")
        self.frame_parameters.rowconfigure((1, 2, 3), weight=1) # configure grid layout of the frame_parameters (6x3)
        self.frame_parameters.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)


        # ============ frame_info (within frame_right) - for DISPLAYING ============
        
        # Label, when no file in the system yet
        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Image file",
                                                   height=100,
                                                   corner_radius=6,
                                                   fg_color=("white", "gray38"),
                                                   justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)
        self.label_info_1.configure(font=("Verdana", 11, "italic"))
        
        # Progressbar, here inactive as no file in the system yet
        self.progressbar = customtkinter.CTkSlider(master=self.frame_info,
                                                   from_ = 1,
                                                   to = 2,
                                                   number_of_steps = 2)                                          
        self.progressbar.grid(row=5, column=0, sticky="ew", padx=15, pady=15)


        # ============ frame_parameters (within frame_right) - for PARAMETERS' ANALYSIS ============
        
        self.button_analysis = customtkinter.CTkButton(master=self.frame_parameters,
                                                text="Vasculature analysis",
                                                command= self.print_parameters)
        self.button_analysis.grid(row=1, column=0, columnspan = 7, pady=10, padx=20)
        
        self.label_diameter = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Diameter [h,v]: ")
        self.label_diameter.grid(row=2, column=0, columnspan=2, pady=2, padx=5, sticky="ew")
        
        self.label_area = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Area: ")
        self.label_area.grid(row=3, column=0, columnspan=2, pady=2, padx=5, sticky="ew")    
        
        self.label_volume = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Volume: ")
        self.label_volume.grid(row=2, column=2, columnspan=2, pady=2, padx=5, sticky="ew")          
        
        self.label_volume_percentage_occupancy = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="V % occupancy: ")
        self.label_volume_percentage_occupancy.grid(row=3, column=2, columnspan=2, pady=2, padx=5, sticky="ew")          
        
        self.label_average_branch_length = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Branch length: ")
        self.label_average_branch_length.grid(row=2, column=4, columnspan=2, pady=2, padx=5, sticky="ew")   

        self.label_number_branches_per_volume = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Nb branches / V: ")
        self.label_number_branches_per_volume.grid(row=3, column=4, columnspan=2, pady=2, padx=5, sticky="ew")   


        # ============ bottom frame (within frame_right) for PLOTTING  ===============
        
        # Choose PARAMETER
        self.combobox_var = customtkinter.StringVar(value="Parameter")
        self.combobox_1 = customtkinter.CTkComboBox(master=self.frame_right,
                                             values=["Branch distance by branch type - only for 1 file, blue color", 
                                                     "Number branches",
                                                     "Number isolated branches",
                                                     "Number endpoint-junction branches",
                                                     "Number junction-junction branches",
                                                     "Number circular branches",
                                                     "Average branch length", 
                                                     "Total length", 
                                                     "Number junctions", 
                                                     "Junctions per volume", 
                                                     "Branches per volume", 
                                                     "Area", 
                                                     "Volume", 
                                                     "V percentage occupancy",
                                                     "Diameter_mean_horizontal",
                                                     "Diameter_mean_vertical",
                                                     "Diameter_horizontal",
                                                     "Diameter_vertical"
                                                     ],
                                             variable=self.combobox_var)
        self.combobox_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")
        
        
        # Choose files = group 1
        self.button_group1 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 1",
                                                     fg_color = "gray12",
                                                     command = self.select_files_group1)
        self.button_group1.grid(row=10, column=1, pady=10, padx=20, sticky="w")
        self.entry_group1 = customtkinter.CTkEntry(master=self.frame_right, placeholder_text="Name 1")
        self.entry_group1.grid(row=11, column=1, pady=10, padx=20, sticky="w")
        # Choose files = group 2
        self.button_group2 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 2",
                                                     fg_color = "gray12",
                                                     command = self.select_files_group2)
        self.button_group2.grid(row=10, column=2, pady=10, padx=20, sticky="w")
        self.entry_group2 = customtkinter.CTkEntry(master=self.frame_right, placeholder_text="Name 2")
        self.entry_group2.grid(row=11, column=2, pady=10, padx=20, sticky="w")
        # Choose files = group 3
        self.button_group3 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 3",
                                                     fg_color = "gray12",
                                                     command = self.select_files_group3)
        self.button_group3.grid(row=10, column=3, pady=10, padx=20, sticky="w")
        self.entry_group3 = customtkinter.CTkEntry(master=self.frame_right, placeholder_text="Name 3")
        self.entry_group3.grid(row=11, column=3, pady=10, padx=20, sticky="w")
        # Choose files = group 4
        self.button_group4 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 4",
                                                     fg_color = "gray12",
                                                     command = self.select_files_group4)
        self.button_group4.grid(row=10, column=4, pady=10, padx=20, sticky="w")
        self.entry_group4 = customtkinter.CTkEntry(master=self.frame_right, placeholder_text="Name 4")
        self.entry_group4.grid(row=11, column=4, pady=10, padx=20, sticky="w")
        # Choose files = group 5
        self.button_group5 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 5",
                                                     fg_color = "gray12",
                                                     command = self.select_files_group5)
        self.button_group5.grid(row=10, column=5, pady=10, padx=20, sticky="w")
        self.entry_group5 = customtkinter.CTkEntry(master=self.frame_right, placeholder_text="Name 5")
        self.entry_group5.grid(row=11, column=5, pady=10, padx=20, sticky="w")
        # Choose files = group 6
        self.button_group6 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 6",
                                                     fg_color = "gray12",
                                                     command = self.select_files_group6)
        self.button_group6.grid(row=10, column=6, pady=10, padx=20, sticky="w")
        self.entry_group6 = customtkinter.CTkEntry(master=self.frame_right, placeholder_text="Name 6")
        self.entry_group6.grid(row=11, column=6, pady=10, padx=20, sticky="w")
        
        # Title request
        self.entry_group_0 = customtkinter.CTkEntry(master=self.frame_right, placeholder_text="Title")
        self.entry_group_0.grid(row=11, column=0, pady=10, padx=20, sticky="w")
        
        

        # ============ DEFAULT VALUES ============
        self.optionmenu_1.set("Dark")
        self.combobox_time.set("Timepoint")
        self.combobox_channel.set("Channel")
        self.combobox_nb_lif.set("Series")
        self.combobox_segmentation.set("Segmentation")
        self.combobox_1.set("Data")
        self.progressbar.set(1)








            








# ========================================================================= FUNCTIONS ==================================================================   




    # ================= OPENING the file ==================
    # In this step, the file (.lif, .tif or .tiff) is opened by the user.
    # The image (a z-stack) is extracted and processed.
    # The meshes are computed.
    # The max projection is displayed.


    def open_file(self):
        
        # ============ Variables ============
        # These variables will be used in other functions
        # Next to each is written in a comment where else this variable is used.
        # FILE
        global time # in save_real_mesh 
        global channel_fluo # in save_real_mesh 
        global nb_lif # in save_real_mesh 
        global segmentation # used in batch processing
        global filename # file path
        global image # for area_volume_diameters
        # PROGRESS
        global process # to know for the error message for save parameters
        global show_var # progressbar
        # ANALYSIS
        global scale_x_px, scale_y_px, scale_z_px, scale_x, scale_y, scale_z, x_total, y_total, z_stacks, x_total_um, y_total_um, depth  # saved in branch_data
        global th, mask, mask_inv, mask_skel, mask_skel_inv # for print_parameters
        global vol, surf, vol_inv_um, voxels, surf_gaussian # for volume, area, mesh saving
        # DISPLAY
        global max_proj, max_proj_toplot # in slider_show, among others
        
        
        
        # ============ OPEN the file ============
        process = 0
        filename = filedialog.askopenfilename(title =  "Select A File", filetypes = (("all files", "*.*"),("lif files", "*.lif"),("tif files", "*.tif"), ("tiff files", "*.tiff")))  # returns the name of the file, and the location of the file
        timepoint = self.combobox_time.get()
        channel = self.combobox_channel.get()
        nb = self.combobox_nb_lif.get()
        segmentation = self.combobox_segmentation.get()
        # Error box
        if timepoint == 'Timepoint' or channel == 'Channel' or nb == 'Series' or segmentation == 'Segmentation':
            self.message = messagebox.showinfo(title = "Error", message = "Please check your timepoint, series, channel and segmentation. \n" +
                                                                          "\n" +
                                                                          "Each of the comboboxes should be filled. \n" +
                                                                          "If there is no timepoint, no channel, please put 0. \n" +
                                                                          "\n" +
                                                                          "The series correspond to the series number, \n" +
                                                                          "associated to each .tif in the .lif, \n" +
                                                                          "the same ones as when opening with ImageJ. \n" +
                                                                          "\n" +
                                                                          "The segmentation can be performed via Otsu thresholding (on) \n" +
                                                                          "or the mask can be given directly (off). \n" +
                                                                          "\n" +
                                                                          "The mask should be in .lif or .tif format, \n" +
                                                                          "in order to have the metadata with it.")
        
        
        
        # ============ IMAGE PROCESSING ============
        else: 
            
                        
        # ============ RENEW the buttons ============            
            
            self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                    text="Choose Other",
                                                    command=self.open_file)
            self.button_1.grid(row=6, column=0, pady=10, padx=20)
            

            self.button_2 = customtkinter.CTkButton(master=self.frame_left, # RENEW the save_parameters button to original state, as if the user had been saving another file right before, it would be in the "SAVED" state
                                                    text="Save parameters",
                                                    command=self.save_parameters)
            self.button_2.grid(row=7, column=0, pady=10, padx=20)

            self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                    text="Save mesh CFD",
                                                    command=self.save_mesh_CFD)
            self.button_3.grid(row=8, column=0, pady=10, padx=20)
            

            self.button_analysis = customtkinter.CTkButton(master=self.frame_parameters, # RENEW the vasculature parameters button
                                                           text="Vasculature analysis",
                                                           command= self.print_parameters)
            self.button_analysis.grid(row=1, column=0, columnspan = 6, pady=10, padx=20)
            
            self.label_diameter = customtkinter.CTkLabel(master=self.frame_parameters,
                                                         text="Diameter [h,v]: ")
            self.label_diameter.grid(row=2, column=0, columnspan=2, pady=2, padx=5, sticky="ew")
            
            self.label_area = customtkinter.CTkLabel(master=self.frame_parameters,
                                                            text="Area: ")
            self.label_area.grid(row=3, column=0, columnspan=2, pady=2, padx=5, sticky="ew")    
            
            self.label_volume = customtkinter.CTkLabel(master=self.frame_parameters,
                                                            text="Volume: ")
            self.label_volume.grid(row=2, column=2, columnspan=2, pady=2, padx=5, sticky="ew")          
            
            self.label_volume_percentage_occupancy = customtkinter.CTkLabel(master=self.frame_parameters,
                                                                            text="% Occupancy: ")
            self.label_volume_percentage_occupancy.grid(row=3, column=2, columnspan=2, pady=2, padx=5, sticky="ew")          
            
            self.label_average_branch_length = customtkinter.CTkLabel(master=self.frame_parameters,
                                                                      text="Branch length: ")
            self.label_average_branch_length.grid(row=2, column=4, columnspan=2, pady=2, padx=5, sticky="ew")   
            
            self.label_number_branches_per_volume = customtkinter.CTkLabel(master=self.frame_parameters,
                                                                         text="Nb branches / V: ")
            self.label_number_branches_per_volume.grid(row=3, column=4, columnspan=2, pady=2, padx=5, sticky="ew")    
            
        
            self.button_group1 = customtkinter.CTkButton(master = self.frame_right,  # RENEW the groups Buttons
                                                         text = "Group 1",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group1)
            self.button_group1.grid(row=10, column=1, pady=10, padx=20, sticky="w")
            self.button_group2 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 2",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group2)
            self.button_group2.grid(row=10, column=2, pady=10, padx=20, sticky="w")
            self.button_group3 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 3",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group3)
            self.button_group3.grid(row=10, column=3, pady=10, padx=20, sticky="w")
            self.button_group4 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 4",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group4)
            self.button_group4.grid(row=10, column=4, pady=10, padx=20, sticky="w")
            self.button_group5 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 5",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group5)
            self.button_group5.grid(row=10, column=5, pady=10, padx=20, sticky="w")
            self.button_group6 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 6",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group6)
            self.button_group6.grid(row=10, column=6, pady=10, padx=20, sticky="w")
            
            
            show_var = tkinter.DoubleVar() # Progressbar (appears when max-projection displayed) 
            show_var.set(1)
            self.progressbar = customtkinter.CTkSlider(master=self.frame_right,
                                                       from_ = 0,
                                                       to = 2,
                                                       number_of_steps = 2,
                                                       variable = show_var,
                                                       command = lambda event: self.slider_show(filename, show_var))
            self.progressbar.grid(row=4, column=0, columnspan = 12, sticky="ew", padx=15, pady=15)
            self.progressbar.set(1)
            
            
            # Extract the image
            time = int(timepoint)
            channel_fluo = int(channel)
            nb_lif = int(nb)-1
            image = extract_z_stack(self, filename, nb_lif, time, channel_fluo)
            scale_x_px, scale_y_px, scale_z_px, scale_x, scale_y, scale_z, x_total, y_total, z_stacks, x_total_um, y_total_um, depth = metadata(filename, nb_lif)
        
            # Image processing
            kernel = 0.5 # to blur just enough for a segmentation as accurate as possible 
            mask, mask_inv, mask_skel, mask_skel_inv = who_is_the_mask(image, segmentation, kernel)
            
            # Make already the meshes, to save time
            vol, surf, vol_inv_um, voxels = real_mesh(mask, mask_inv, scale_x, scale_y, scale_z)
            
            
            
        # ============ DISPLAY max projection ============
            max_proj, max_proj_toplot = max_projection(image, x_total, y_total, x_total_um, y_total_um)
            
            # Frame in which to show the max-projection
            self.frame_info = customtkinter.CTkFrame(master=self.frame_right, bg_color = 'gray18')
            self.frame_info.grid(row=0, column=0, columnspan=7, rowspan=4, pady=20, padx=20, sticky="nsew")
            self.frame_inside_info = customtkinter.CTkFrame(master=self.frame_info, bg_color = 'gray18')
            self.frame_inside_info.pack(side=tkinter.TOP, fill=tkinter.BOTH)#, expand=0.8)
            
            # DISPLAY
            fig = plt.Figure(figsize=(6,5), dpi=100)
            fig.patch.set_facecolor('black')
            ax = fig.add_subplot()
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.tick_params(labelcolor='white')
            ax.tick_params(direction='out', length=6, width=0.5, colors='white')
            ax.set_ylabel('Length (µm)', color='white')
            # In a Canvas
            canvas = FigureCanvasTkAgg(fig, self.frame_inside_info)
            canvas.get_tk_widget().configure(width=250, height=250)
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
            # Create Matplotlib Toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.frame_inside_info, pack_toolbar=False)
            toolbar.config(background='ivory')
            toolbar._message_label.config(background='ivory')
            toolbar.update()
            toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
            # Show
            ax.imshow(max_proj_toplot)
            canvas.draw()
            
            









    # ================= ANALYSING the vasculature ==================
    # In this step, the skeleton is computed and analysed.
    # The morphological features (area, volume, diameters) are extracted in 3D.


    def print_parameters(self):
        
        # ============ Variables ============
        global process
        global area, volume, volume_total_fibrin_gel, area_total_fibrin_gel, volume_percentage_occupancy
        global branch_data, number_branches, number_isolated_branches, number_endpoint_junctions_branches, number_junction_junction_branches, number_circular_branches, number_junctions, number_endpoints, average_branch_length, total_length
        global junction_per_volume, number_branches_per_volume
        global diameter, diameter_horizontal, diameter_vertical, mean_diameter_horizontal, mean_diameter_vertical
        global skeleton_3d # for saving the x, y ,z of the skeleton
        
        process = 1
        
        
        # ============ SKELETON analysis ============
        skeleton_3d = skeleton_3d_make(mask_skel_inv)
        sk_3d_projection_clean, branch_data, number_branches, number_isolated_branches, number_endpoint_junctions_branches, number_junction_junction_branches, number_circular_branches, number_junctions, number_endpoints, average_branch_length, total_length, endpoints, junctions  = skeleton_3d_analysis(skeleton_3d, x_total, y_total, scale_x)
        coord_src_0, coord_src_1, coord_dst_0, coord_dst_1 = summarize_props_skeleton(branch_data)
        # POP-UP to show the skeleton - for the user's checking purposes
        img = max_proj.copy()
        y,x = sk_3d_projection_clean.nonzero() # skeleton on mask
        a=[]
        b=[]
        c=[]
        d=[]
        for i in range (len(endpoints)):
            b.append(endpoints[i][0])
            a.append(endpoints[i][1])
        for j in range (len(junctions)):
            d.append(junctions[j][0])
            c.append(junctions[j][1]) 
        fig, ax = plt.subplots()
        ax.scatter(x, y, s = 3, color='black')
        ax.scatter(a, b, s= 20, color = 'fuchsia') # endpoints
        ax.scatter(c, d, s = 20, color='aqua') # junctions
        plt.suptitle('Skeleton of the file < ' + Path(filename).stem +' >, and its junctions (blue) and endpoints (pink) ')
        plt.imshow(img)
        
        # ============ MORPHOLOGICAL analysis: area, volume, diameters ============
        area, volume, volume_total_fibrin_gel, area_total_fibrin_gel, volume_percentage_occupancy, diameter_horizontal, diameter_vertical, mean_diameter_horizontal, mean_diameter_vertical = area_volume_diameters(self, surf, voxels, image, mask_skel, scale_x, scale_y, scale_z, x_total_um, y_total_um, depth, coord_src_0, coord_dst_0, coord_src_1, coord_dst_1)

        # ============ SKELETON analysis: per area ============
        junction_per_volume = number_junctions/volume
        number_branches_per_volume = number_branches / volume
        
        
        
        # ============ DISPLAY the results ============
        self.button_analysis = customtkinter.CTkButton(master=self.frame_parameters,
                                                text="Vasculature analysis",
                                                state = tkinter.DISABLED)
        self.button_analysis.grid(row=1, column=0, columnspan = 6, pady=10, padx=20)
        
        self.label_diameter = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Diameter [h,v]: " + f"{mean_diameter_horizontal:.1E}" + " µm, " + f"{mean_diameter_vertical:.1E}"  + " µm")
        self.label_diameter.grid(row=2, column=0, columnspan=2, pady=2, padx=5, sticky="ew")
        
        self.label_area = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Area: " + f"{area:.1E}" + " µm2")
        self.label_area.grid(row=3, column=0, columnspan=2, pady=2, padx=5, sticky="ew")  
        
        self.label_volume = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Volume: " + f"{volume:.1E}" + " µm3")
        self.label_volume.grid(row=2, column=2, columnspan=2, pady=2, padx=5, sticky="ew")       
        
        self.label_volume_percentage_occupancy = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="% Occupancy: " + f"{round(volume_percentage_occupancy*100,1)}" + " %")
        self.label_volume_percentage_occupancy.grid(row=3, column=2, columnspan=2, pady=2, padx=5, sticky="ew")          
        
        self.label_average_branch_length = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Branch length: " + f"{average_branch_length:.1E}" + " µm")
        self.label_average_branch_length.grid(row=2, column=4, columnspan=2, pady=2, padx=5, sticky="ew")   

        self.label_number_branches_per_volume = customtkinter.CTkLabel(master=self.frame_parameters,
                                                        text="Nb branches / V: " + f"{number_branches_per_volume:.1E}")
        self.label_number_branches_per_volume.grid(row=3, column=4, columnspan=2, pady=2, padx=5, sticky="ew")  
        






    # ================= SAVING the results of the analysis ==================
    # Here, the results of the 3D analysis are saved in an excel sheet.
    
    def save_parameters(self):
          if process == 0:
              messagebox.showerror("Error", "Please analyse the vasculature") 
          else:
              self.message = messagebox.showinfo(title = "Excel directory", message = "Please choose your excel file location")
              xcl_directory = filedialog.askdirectory()
              
              # Add the rest of the parameters to branch_data
              branch_data['number_branches'] = pd.Series(number_branches)
              branch_data['number_isolated_branches'] = pd.Series(number_isolated_branches)
              branch_data['number_endpoint_junctions_branches'] = pd.Series(number_endpoint_junctions_branches)
              branch_data['number_junction_junction_branches'] = pd.Series(number_junction_junction_branches)
              branch_data['number_circular_branches'] = pd.Series(number_circular_branches)
              branch_data['number_junctions'] = pd.Series(number_junctions)
              branch_data['number_endpoints'] = pd.Series(number_endpoints)
              branch_data['average_branch_length'] = pd.Series(average_branch_length)
              branch_data['total_length'] = pd.Series(total_length)
              branch_data['junction_per_volume'] = pd.Series(junction_per_volume)
              branch_data['branches_per_volume']=pd.Series(number_branches_per_volume)

              branch_data['area_fibrin'] = pd.Series([area_total_fibrin_gel])
              branch_data['area_vessels'] = pd.Series([area])
              branch_data['volume_fibrin'] = pd.Series([volume_total_fibrin_gel])
              branch_data['volume_vessels'] = pd.Series([volume])
              branch_data["v_percentage_occupancy"] = pd.Series([volume_percentage_occupancy])

              branch_data['diameters_horizontal'] = pd.Series(diameter_horizontal)
              branch_data['diameters_vertical'] = pd.Series(diameter_vertical)
              branch_data['mean diameters_horizontal'] = pd.Series([mean_diameter_horizontal])
              branch_data['mean diameters_vertical'] = pd.Series([mean_diameter_vertical])
          
              branch_data['metadata_id'] = pd.Series(['scale_x_px', 'scale_y_px', 'scale_z_px', 'scale_x', 'scale_y', 'scale_z', 'x_total', 'y_total', 'z_stacks', 'x_total_um', 'y_total_um', 'depth'])
              branch_data['metadata'] = pd.Series([scale_x_px, scale_y_px, scale_z_px, scale_x, scale_y, scale_z, x_total, y_total, z_stacks, x_total_um, y_total_um, depth])
              
              # save x, y, z of the skeleton
              xyz_skel = pd.DataFrame()
              z_skel, y_skel, x_skel = skeleton_3d.nonzero()
              z_skel_um = z_skel*scale_z
              y_skel_um = y_skel*scale_y
              x_skel_um = x_skel*scale_x
              xyz_skel['x'] = x_skel_um
              xyz_skel['y'] = y_skel_um
              xyz_skel['z'] = z_skel_um
              
              # save branch_data
              file_name_for_xcl = Path(filename).stem + '_data_analysis series n°' +str(nb_lif+1) + ' t=' + str(time) + ' c=' + str(channel_fluo)
              xcl_path = str(xcl_directory + '/' + file_name_for_xcl + ".xlsx")
              
              # save x, y, z of the skeleton
              file_name_for_xcl_skel = Path(filename).stem + '_data_analysis series n°' +str(nb_lif+1) + ' t=' + str(time) + ' c=' + str(channel_fluo) +' skeleton x, y ,z'
              xcl_path_skel = str(xcl_directory + '/' + file_name_for_xcl_skel + ".xlsx")              
              
              
              if xcl_directory:
                  branch_data.to_excel(xcl_path, index=False) 
                  xyz_skel.to_excel(xcl_path_skel, index=False)
              
                  # ============ RENEW the buttons ============         
                  # If saved, renew the save_parameters button
                  self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                          text="SAVED",
                                                          state = tkinter.DISABLED)
                  self.button_2.grid(row=7, column=0, pady=10, padx=20)
   
    
    
   
    
   
    # ================= SAVING the CFD mesh ================== 
    # After tests on the Comsol platform,
    # The best simplified mesh was designed with the following steps:
    # processing for correct size for comsol: gaussian5 + decimate0.01 + largest connected region + triangulate.
    # These steps enable the user to open the file on a normal computer, not needing too much memory,
    # and ensures simplified processing on comsol,
    # while preserving the orientation, diameters, network of the microvessels.
    
   
    def save_mesh_CFD(self):
        global surf2
        stl_directory = filedialog.askdirectory()
        file_name_for_stl = Path(filename).stem + '_for_CFD_series n°' +str(nb_lif+1) + ' t=' + str(time) + ' c=' + str(channel_fluo)

        if stl_directory: 
            stl_path = str(stl_directory + '/' + file_name_for_stl + ".stl")
            
            img_equal_CFD = exposure.equalize_adapthist(image, clip_limit=0.03) 
            im_blur_CFD = gaussian_filter(img_equal_CFD, sigma=5)
            th_CFD = threshold_otsu(im_blur_CFD)
            mask_CFD = np.array(im_blur_CFD < th_CFD, dtype=np.int8)
            mask_transposed_CFD = np.transpose(mask_CFD, axes = [2,1,0])
            vol_CFD = vedo.Volume(mask_transposed_CFD, spacing=(scale_x,scale_y,scale_z)) # µm/px
            surf_CFD=vol_CFD.isosurface()

            surf2_CFD = surf_CFD.decimate(fraction=0.01, N=None, method='quadric', boundaries=False) 
            surf2_CFD.triangulate()
            surf2_CFD.extractLargestRegion()

            surf2_CFD.write(stl_path)
            
        # ============ RENEW the button ============      
        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                              text="SAVED",
                                              state = tkinter.DISABLED)
        self.button_3.grid(row=8, column=0, pady=10, padx=20)







    # ================= SHOWING either the mac projection, or the real mesh ================== 
    
    def slider_show(self, filename, show_var):
        global canvas # otherwise did not work the next time for the displaying
        var_progressbar = int(show_var.get())
        
        if filename is not None:
            if var_progressbar==1 or var_progressbar == 0:
                # Frame in which to show the max projection
                self.frame_info = customtkinter.CTkFrame(master=self.frame_right, bg_color = 'gray18')
                self.frame_info.grid(row=0, column=0, columnspan=7, rowspan=4, pady=20, padx=20, sticky="nsew")
                self.frame_info.rowconfigure(0, weight=1)
                self.frame_info.columnconfigure(0, weight=1)
                self.frame_inside_info = customtkinter.CTkFrame(master=self.frame_info, bg_color = 'gray18')
                self.frame_inside_info.grid(row = 0, column = 0, pady = 20, padx = 20, sticky = "nsew")
                self.frame_inside_info.rowconfigure((0, 1, 2, 3), weight=1)
                self.frame_inside_info.columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)    
                
                # Set figure appearance
                fig = plt.Figure(figsize=(6,5), dpi=100)
                fig.patch.set_facecolor('black')
                ax = fig.add_subplot()
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.tick_params(labelcolor='white')
                ax.tick_params(direction='out', length=6, width=0.5, colors='white', )
                ax.set_ylabel('Length (µm)', color='white')
                # In a Canvas
                canvas = FigureCanvasTkAgg(fig, self.frame_inside_info)
                canvas.draw()
                # Create Matplotlib Toolbar
                toolbar = NavigationToolbar2Tk(canvas, self.frame_inside_info, pack_toolbar=False)
                toolbar.config(background='ivory')
                toolbar._message_label.config(background='ivory')
                toolbar.update()
                #toolbar.grid(row=4, column=0, pady=20, padx=20, columnspan = 7)
                toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
                # Show
                ax.imshow(max_proj_toplot)
                canvas.get_tk_widget().configure(width=300, height=300)
                canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH)#, expand=True)
                
            elif var_progressbar == 2:
                self.frame_info = customtkinter.CTkFrame(master=self.frame_right, bg_color = 'gray18')
                self.frame_info.grid(row=0, column=0, columnspan=7, rowspan=4, pady=20, padx=20, sticky="nsew")
                self.frame_info.rowconfigure(0, weight=1)
                self.frame_info.columnconfigure(0, weight=1)
                self.frame_inside_info = customtkinter.CTkFrame(master=self.frame_info, bg_color = 'gray18')
                self.frame_inside_info.grid(row = 0, column = 0, pady = 20, padx = 20, sticky = "nsew")
                self.frame_inside_info.rowconfigure(0, weight=1)
                self.frame_inside_info.rowconfigure(2, weight=1)
                self.frame_inside_info.columnconfigure(0, weight=1)
                self.frame_inside_info.columnconfigure(2, weight=1)
                # ============ RENEW the button ============      
                self.button_4 = customtkinter.CTkButton(master = self.frame_inside_info,
                                                        text = "Save - SHOW real mesh",
                                                        text_font=("Roboto Medium", -20),
                                                        fg_color = "dodger blue",
                                                        width=300,
                                                        height=100,
                                                        corner_radius=10,
                                                        command = self.save_real_mesh_plot)
                self.button_4.grid(row=1, column=1, pady=10, padx=20)
                if canvas:
                    canvas.get_tk_widget().delete("all")
                    canvas = canvas.get_tk_widget().destroy()
                    canvas = None
                    
  
    
  
    
  
    
    # ================= SHOWING the real mesh ==================  
    # The real mesh is surf.
    # The user does not have to input any file name, the saving is automatically 
    # performed in the same way, so that there is no messing up when writing the name,
    # and so that all the info characterizing the file is present in the title.

               
    def save_real_mesh_plot(self):
        stl_directory = filedialog.askdirectory()
        file_name_for_stl = Path(filename).stem + '_real_mesh' + '_series ' + str(nb_lif) + ' t=' + str(time) + ' c=' + str(channel_fluo)

        if stl_directory: 
            stl_path = str(stl_directory + '/' + file_name_for_stl + ".stl")
            surf_connected = surf.extract_largest_region() # surf = real mesh, extract largest region to have a nice, clean mesh
            surf_connected.write(stl_path)
            
            # Display the real mesh
            mesh_real = pv.read(stl_path)   
            plotter = pq.BackgroundPlotter()
            plotter.add_mesh(mesh_real, color='moccasin', metallic=1.0)
            plotter.add_axes()
            plotter.add_bounding_box()
            plotter.show_bounds(padding=0.4)
            plotter.set_background('black')
            plotter.show() 
            
            self.frame_info = customtkinter.CTkFrame(master=self.frame_right, bg_color = 'gray18')
            self.frame_info.grid(row=0, column=0, columnspan=7, rowspan=4, pady=20, padx=20, sticky="nsew")
            self.frame_info.rowconfigure(0, weight=1)
            self.frame_info.columnconfigure(0, weight=1)
            self.frame_inside_info = customtkinter.CTkFrame(master=self.frame_info, bg_color = 'gray18')
            self.frame_inside_info.grid(row = 0, column = 0, pady = 20, padx = 20, sticky = "nsew")
            self.frame_inside_info.rowconfigure(0, weight=1)
            self.frame_inside_info.rowconfigure(2, weight=1)
            self.frame_inside_info.columnconfigure(0, weight=1)
            self.frame_inside_info.columnconfigure(2, weight=1)
            # ============ RENEW the button ============      
            self.button_4 = customtkinter.CTkButton(master = self.frame_inside_info,
                                                    text = "SAVED",
                                                    text_font=("Roboto Medium", -20),
                                                    fg_color = "dodger blue",
                                                    width=300,
                                                    height=100,
                                                    corner_radius=10,
                                                    command = self.save_real_mesh_plot)
            self.button_4.grid(row=1, column=1, pady=10, padx=20)
    
        



    # ================= BATCH processing ==================  
    # The files can be processed in batch.
    # This is not recommended when segmentation is put on "on", 
    # as the user should check every time if the segmentation was performed correctly.
    
    
    def batch(self):
        
        # 1. Choose the location of the excel file
        self.message = messagebox.showinfo(title = "Excel directory", message = "Please choose your excel file location")
        xcl_directory = filedialog.askdirectory()
        
        # 2. Choose the files to analyse
        if xcl_directory:
            self.message_2 = messagebox.showinfo(title = "File directory", message = "Please choose your file(s) to analyze")
            file = filedialog.askopenfilenames(title =  "Select A File", filetypes = (("all files", "*.*"),("lif files", "*.lif"),("tif files", "*.tif"), ("tiff files", "*.tiff")))
            
            # 3. The processing and analyses are performed in batch.
            if file:
                timepoint = self.combobox_time.get()
                channel = self.combobox_channel.get()
                segmentation = self.combobox_segmentation.get()
                if timepoint == 'Timepoint' or channel == 'Channel' or segmentation == 'Segmentation':
                    self.message = messagebox.showinfo(title = "Error", message = "Please check your timepoint, series channel and segmentation. \n" +
                                                                                  "\n" +
                                                                                  "Each of the comboboxes should be filled. \n" +
                                                                                  "If there is no timepoint, no channel, please put 0. \n" +
                                                                                  "\n" +
                                                                                  "Please put 1 for the SERIES in batch processing \n" +
                                                                                  "\n" +
                                                                                  "The segmentation can be performed via Otsu thresholding (on) \n" +
                                                                                  "or the mask can be given directly (off). \n" +
                                                                                  "\n" +
                                                                                  "The mask should be in .lif or .tif format, \n" +
                                                                                  "in order to have the metadata with it.")
                else:

                    time = int(timepoint)
                    channel_fluo = int(channel)
                    for i in file: # for filename in the file
                        reader = read_lif.Reader(i)
                        series = reader.getSeries() # series of files
                        for j in range (len(series)):
                            chose = series[j] # one file 
                            image = chose.getFrame(T=time, channel=channel_fluo)
                            scale_x_px, scale_y_px, scale_z_px, scale_x, scale_y, scale_z, x_total, y_total, z_stacks, x_total_um, y_total_um, depth = metadata(i, j)
                        
                            # Image processing
                            kernel = 0.5 # to blur just enough for a segmentation as accurate as possible 
                            mask, mask_inv, mask_skel, mask_skel_inv = who_is_the_mask(image, segmentation, kernel)
                            
                            # Make already the meshes, to save time
                            vol, surf, vol_inv_um, voxels = real_mesh(mask, mask_inv, scale_x, scale_y, scale_z)
                            
                            # Skeleton
                            skeleton_3d = skeleton_3d_make(mask_skel_inv)
                            sk_3d_projection_clean, branch_data, number_branches, number_isolated_branches, number_endpoint_junctions_branches, number_junction_junction_branches, number_circular_branches, number_junctions, number_endpoints, average_branch_length, total_length, endpoints, junctions  = skeleton_3d_analysis(skeleton_3d, x_total, y_total, scale_x)
                            coord_src_0, coord_src_1, coord_dst_0, coord_dst_1 = summarize_props_skeleton(branch_data)
                            
                            # Area, Volume, Diameters
                            area, volume, volume_total_fibrin_gel, area_total_fibrin_gel, volume_percentage_occupancy, diameter_horizontal, diameter_vertical, mean_diameter_horizontal, mean_diameter_vertical = area_volume_diameters(self, surf, voxels, image, mask_skel, scale_x, scale_y, scale_z, x_total_um, y_total_um, depth, coord_src_0, coord_dst_0, coord_src_1, coord_dst_1)
                            
                            junction_per_volume = number_junctions/volume
                            number_branches_per_volume = number_branches / volume
                            
                            # Add the rest of the parameters to branch_data
                            branch_data['number_branches'] = pd.Series(number_branches)
                            branch_data['number_isolated_branches'] = pd.Series(number_isolated_branches)
                            branch_data['number_endpoint_junctions_branches'] = pd.Series(number_endpoint_junctions_branches)
                            branch_data['number_junction_junction_branches'] = pd.Series(number_junction_junction_branches)
                            branch_data['number_circular_branches'] = pd.Series(number_circular_branches)
                            branch_data['number_junctions'] = pd.Series(number_junctions)
                            branch_data['number_endpoints'] = pd.Series(number_endpoints)
                            branch_data['average_branch_length'] = pd.Series(average_branch_length)
                            branch_data['total_length'] = pd.Series(total_length)
                            branch_data['junction_per_volume'] = pd.Series(junction_per_volume)
                            branch_data['branches_per_volume']=pd.Series(number_branches_per_volume)
                            
                            branch_data['area_fibrin'] = pd.Series([area_total_fibrin_gel])
                            branch_data['area_vessels'] = pd.Series([area])
                            branch_data['volume_fibrin'] = pd.Series([volume_total_fibrin_gel])
                            branch_data['volume_vessels'] = pd.Series([volume])
                            branch_data["v_percentage_occupancy"] = pd.Series([volume_percentage_occupancy])
                            
                            branch_data['diameters_horizontal'] = pd.Series(diameter_horizontal)
                            branch_data['diameters_vertical'] = pd.Series(diameter_vertical)
                            branch_data['mean diameters_horizontal'] = pd.Series([mean_diameter_horizontal])
                            branch_data['mean diameters_vertical'] = pd.Series([mean_diameter_vertical])
                            
                            branch_data['metadata_id'] = pd.Series(['scale_x_px', 'scale_y_px', 'scale_z_px', 'scale_x', 'scale_y', 'scale_z', 'x_total', 'y_total', 'z_stacks', 'x_total_um', 'y_total_um', 'depth'])
                            branch_data['metadata'] = pd.Series([scale_x_px, scale_y_px, scale_z_px, scale_x, scale_y, scale_z, x_total, y_total, z_stacks, x_total_um, y_total_um, depth])
                              
                            # save branch_data
                            file_name_for_xcl = Path(i).stem + ' batch_data_analysis series n°' +str(j+1) + ' t=' + str(time) + ' c=' + str(channel_fluo)
                            xcl_path = str(xcl_directory + '/' + file_name_for_xcl + ".xlsx")
                            branch_data.to_excel(xcl_path, index=False) 
                            
                            # save x, y, z of the skeleton
                            xyz_skel = pd.DataFrame()
                            z_skel, y_skel, x_skel = skeleton_3d.nonzero()
                            z_skel_um = z_skel*scale_z
                            y_skel_um = y_skel*scale_y
                            x_skel_um = x_skel*scale_x
                            xyz_skel['x'] = x_skel_um
                            xyz_skel['y'] = y_skel_um
                            xyz_skel['z'] = z_skel_um
                            
                            # save x, y, z of the skeleton
                            file_name_for_xcl_skel = Path(i).stem + ' batch_data_analysis series n°' +str(j+1) + ' t=' + str(time) + ' c=' + str(channel_fluo) + ' skeleton x, y ,z'
                            xcl_path_skel = str(xcl_directory + '/' + file_name_for_xcl_skel + ".xlsx")
                            xyz_skel.to_excel(xcl_path_skel, index=False)
                        
                
                        

        
        

    # ============================== STATISTICS and PLOTTING ==================================    
    # The user can choose the groups - a group being a group of files that belong to the same experimental group,
    # Such as 3 files in control, and 3 files in experimental conditions.
    # The user has to choose a title, names for each group, and colors.
    # Then, the statistics are computed for him.
    # Finally, the graph is displayed, and can be saved under different formats.
    
    # ================= SELECT the files ==================  
    
    def select_files_group1(self):
        global filenames_group1
        global files
        files =[]
        filetypes = (('Excel files', '*.xlsx'),('All files', '*.*'))
        filenames_group1 = filedialog.askopenfilenames(
            title='Open files',
            initialdir='/',
            filetypes=filetypes)
        files.append(filenames_group1)
        # Choose files = group 1
        self.button_group1 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 1",
                                                     fg_color = "turquoise1",
                                                     command = self.select_files_group1)
        self.button_group1.grid(row=10, column=1, pady=10, padx=20, sticky="w")
     
    def select_files_group2(self):
        global filenames_group2
        filetypes = (('Excel files', '*.xlsx'),('All files', '*.*'))
        filenames_group2 = filedialog.askopenfilenames(
            title='Open files',
            initialdir='/',
            filetypes=filetypes)    
        files.append(filenames_group2)
        # Choose files = group 2
        self.button_group2 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 2",
                                                     fg_color = "turquoise1",
                                                     command = self.select_files_group2)
        self.button_group2.grid(row=10, column=2, pady=10, padx=20, sticky="w")

    def select_files_group3(self):
        global filenames_group3
        filetypes = (('Excel files', '*.xlsx'),('All files', '*.*'))
        filenames_group3 = filedialog.askopenfilenames(
            title='Open files',
            initialdir='/',
            filetypes=filetypes) 
        files.append(filenames_group3)
        # Choose files = group 3
        self.button_group3 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 3",
                                                     fg_color = "turquoise1",
                                                     command = self.select_files_group3)
        self.button_group3.grid(row=10, column=3, pady=10, padx=20, sticky="w")
        
    def select_files_group4(self):
        global filenames_group4
        filetypes = (('Excel files', '*.xlsx'),('All files', '*.*'))
        filenames_group4 = filedialog.askopenfilenames(
            title='Open files',
            initialdir='/',
            filetypes=filetypes) 
        files.append(filenames_group4)
        # Choose files = group 4
        self.button_group4 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 4",
                                                     fg_color = "turquoise1",
                                                     command = self.select_files_group4)
        self.button_group4.grid(row=10, column=4, pady=10, padx=20, sticky="w")
                
    def select_files_group5(self):
        global filenames_group5
        filetypes = (('Excel files', '*.xlsx'),('All files', '*.*'))
        filenames_group5 = filedialog.askopenfilenames(
            title='Open files',
            initialdir='/',
            filetypes=filetypes)   
        files.append(filenames_group5)
        # Choose files = group 5
        self.button_group5 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 5",
                                                     fg_color = "turquoise1",
                                                     command = self.select_files_group5)
        self.button_group5.grid(row=10, column=5, pady=10, padx=20, sticky="w")
                
    def select_files_group6(self):
        global filenames_group6
        filetypes = (('Excel files', '*.xlsx'),('All files', '*.*'))
        filenames_group6 = filedialog.askopenfilenames(
            title='Open files',
            initialdir='/',
            filetypes=filetypes)  
        files.append(filenames_group6)
        # Choose files = group 6
        self.button_group6 = customtkinter.CTkButton(master = self.frame_right, 
                                                     text = "Group 6",
                                                     fg_color = "turquoise1",
                                                     command = self.select_files_group6)
        self.button_group6.grid(row=10, column=6, pady=10, padx=20, sticky="w")
        
        
        
        
        
    # ================= READ the analysis data ==================   
    
    def read_excel(self, group_files, parameter): # group_files = files of the group 
        full_data = []    
        for sheet in group_files:
            df = pd.read_excel(sheet)
            
            if parameter == "Number branches":
                location = df.loc[df["number_branches"].notna()] # create a dataframe to avoid the Nan values
                data = location.loc[:, "number_branches"] # locate the values of the desired column
                data = data.to_numpy()
                name = "Number branches"
            elif parameter == "Number isolated branches":
                location = df.loc[df["number_isolated_branches"].notna()]
                data = location.loc[:, "number_isolated_branches"]
                data = data.to_numpy()
                name = "Number isolated branches"
            elif parameter == "Number endpoint-junction branches":
                location = df.loc[df["number_endpoint_junctions_branches"].notna()]
                data = location.loc[:, "number_endpoint_junctions_branches"]
                data = data.to_numpy()
                name = "Number endpoint-junction branches"
            elif parameter == "Number junction-junction branches":
                location = df.loc[df["number_junction_junction_branches"].notna()]
                data = location.loc[:, "number_junction_junction_branches"]
                data = data.to_numpy()
                name = "Number junction-junction branches"
            elif parameter == "Number circular branches":
                location = df.loc[df["number_circular_branches"].notna()]
                data = location.loc[:, "number_circular_branches"]
                data = data.to_numpy()
                name = "Number circular branches"
            elif parameter == "Number junctions":
                location = df.loc[df["number_junctions"].notna()]
                data = location.loc[:, "number_junctions"]
                data = data.to_numpy()
                name = "Number junctions"                    
            elif parameter == "Number endpoints":
                location = df.loc[df["number_endpoints"].notna()]
                data = location.loc[:, "number_endpoints"]
                data = data.to_numpy()
                name = "Number endpoints"
            elif parameter == "Average branch length":
                location = df.loc[df["average_branch_length"].notna()]
                data = location.loc[:, "average_branch_length"]
                data = data.to_numpy()
                name = "Average branch length (µm)"
            elif parameter == "Total length":
                location = df.loc[df["total_length"].notna()]
                data = location.loc[:, "total_length"]
                data = data.to_numpy()
                name = "Total length (µm)"
            elif parameter == "Junctions per volume":
                location = df.loc[df["junction_per_volume"].notna()]
                data = location.loc[:, "junction_per_volume"]
                data = data.to_numpy()
                name = "Junctions per volume (µm-3)"
            elif parameter == "Branches per volume":
                location = df.loc[df["branches_per_volume"].notna()]
                data = location.loc[:, "branches_per_volume"]
                data = data.to_numpy()
                name = "Branches per volume (µm-3)"
            elif parameter == "Area":
                location = df.loc[df["area_vessels"].notna()]
                data = location.loc[:, "area_vessels"]
                data = data.to_numpy()
                name = "Area (µm2)"
            elif parameter == "Volume":
                location = df.loc[df["volume_vessels"].notna()]
                data = location.loc[:, "volume_vessels"]
                data = data.to_numpy()
                name = "Volume (µm3)"
            elif parameter == "V percentage occupancy":
                location = df.loc[df["v_percentage_occupancy"].notna()]
                data = location.loc[:, "v_percentage_occupancy"]
                data = data.to_numpy()
                name = "V percentage occupancy (%)"
                
            elif parameter == "Diameter_mean_horizontal":
                location = df.loc[df["mean diameters_horizontal"].notna()]
                data = location.loc[:, "mean diameters_horizontal"]
                data = data.to_numpy()
                name = "Diameters (µm)"
            elif parameter == "Diameter_mean_vertical":
                location = df.loc[df["mean diameters_vertical"].notna()]
                data = location.loc[:, "mean diameters_vertical"]
                data = data.to_numpy()
                name = "Diameters (µm)"
    
                    
            # Multiple values
            elif parameter == "Diameter_horizontal":
                location = df.loc[df["diameters_horizontal"].notna()]
                data = location.loc[:, "diameters_horizontal"]
                data = data.to_numpy()
                name = "Diameters (µm)"
            elif parameter == "Diameter_vertical":
                location = df.loc[df["diameters_vertical"].notna()]
                data = location.loc[:, "diameters_vertical"]
                data = data.to_numpy()
                name = "Diameters (µm)"
            for i in data:
                full_data.append(i)

        return full_data, name
    
    
    
    
    # ================= Check the statistical assumptions ==================  
    # The statistical assumptions are independence, normality, homoscedasticity.
    
    def check_assumptions(self, data_for_statistics):
        # INDEPENDENCE
        text0 = "The independence has to be checked by the user"
        
        # NORMALITY
        normality = []
        for x in data_for_statistics: # we compute the normality on each group studied
            res = kstest(x, normaldensityfunction.cdf)
            normality.append(res[1]) # extract the p-value
        check_normality = all(x > 0.05 for x in normality)
        if check_normality is True:
            text1 = "Kolmogorov-Smirnov test indicates that the Normality is respected for all the samples" 
        else:
            text1 = "Kolmogorov-Smirnov test indicates that the Normality is NOT respected for all the samples"
            
        # HOMOSCEDASTICITY
        res, p_var = levene(*data_for_statistics)# usually, samples are small. We thus use Levene's test. Note the use of * in the call that unpacks the list, which is the same as calling with multiple argument
        if p_var > 0.05:
            text2 = "Levene's test indicates equal variances between populations"
            check_homoscedasticity = True
        else:
            text2 = "Levene's test indicates that the variances DIFFER between populations"
            check_homoscedasticity = False
        
        return text0, text1, text2, check_normality, check_homoscedasticity



    # ================= STATISTICS computing ==================  
    
    def statistics(self, group_name, data_for_statistics, data_parameter, check_normality, check_homoscedasticity):
        global statistics_name, p_value, number_samples # used in violin_plot
        
        
        # ======== 2 groups =========
        
        if len(data_for_statistics) == 2: # t-test between 2 independent groups
            x_list = [0,1] # list of the columns on which add the stats annotations
            
            # PARAMETRIC test
            if check_normality is True and check_homoscedasticity is True:
                statistics_name = "Student's t-test"
                statistic, p_value = ttest_ind(data_for_statistics[0], data_for_statistics[1])
                if 0.01 < p_value < 0.05:
                    label_stat = "*"
                elif 0.001 < p_value < 0.01:
                    label_stat = "**"
                elif p_value < 0.001:
                    label_stat = "***"
                else:
                    label_stat = "ns (p>0.05)"
                y, h, col = data_parameter['Data'].max() + 30, 20, 'k' # determine the height of the annotations    
                plt.plot([x_list[0], x_list[0], x_list[1], x_list[1]], [y, y+h, y+h, y], lw=1.5, c=col)
                plt.text((x_list[0]+x_list[1])*.5, y+h, label_stat, ha='center', va='bottom', color=col)
                p_value = f"{p_value:.1E}"
                    
            # NON PARAMETRIC test
            else:
                statistics_name = "Mann-Whitney U test"
                statistic, p_value = mannwhitneyu(data_for_statistics[0], data_for_statistics[1]) # to cross-check the statistic: https://planetcalc.com/7858/
                if 0.01 < p_value < 0.05:
                    label_stat = "*"
                elif 0.001 < p_value < 0.01:
                    label_stat = "**"
                elif p_value < 0.001:
                    label_stat = "***"
                else:
                    label_stat = "ns (p>0.05)"
            y, h, col = data_parameter['Data'].max() + 30, 20, 'k' # determine the height of the annotations    
            plt.plot([x_list[0], x_list[0], x_list[1], x_list[1]], [y, y+h, y+h, y], lw=1.5, c=col)
            plt.text((x_list[0]+x_list[1])*.5, y+h, label_stat, ha='center', va='bottom', color=col)
            p_value = f"{p_value:.1E}"
           
            
           
        # ========= More than 2 groups =========  
        
        elif len(data_for_statistics) > 2:
            
            # PARAMETRIC test
            if check_normality is True and check_homoscedasticity is True:
                statistics_name = "One-way ANOVA"
                statistic, pvalue = f_oneway(*data_for_statistics)
                label_stat = []
                x_list = []
                p_value = []
                
                # post-hoc test
                if pvalue < 0.05: 
                    tukey = tukey_hsd(*data_for_statistics)
                    for i in range (len(tukey.pvalue)):
                        p_list_i = tukey.pvalue[i] # tukey needs normality and homoscedasticity
                        # In case of unequal sample sizes, the tukey test uses the Tukey-Kramer method
                        for j in range (len(p_list_i)):
                            pval = p_list_i[j] # p_value for the comparison between groups i and j.
                            p_value.append(f"{pval:.1E}")
                            if 0.01 < pval < 0.05:
                                label_stat = "*"
                                y, h, col = data_parameter['Data'].max() + 30 + k, 10, 'k' # determine the height of the annotations
                                plt.plot([i, j], [y+h, y+h], lw=1.5, c=col) # i first group, j second group
                                plt.text((i+j)*.5, y+h, label_stat, ha='center', va='bottom', color=col)
                                k+=20 # for plotting
                            elif 0.001 < pval < 0.01:
                                label_stat = "*"
                                y, h, col = data_parameter['Data'].max() + 30 + k, 10, 'k' # determine the height of the annotations
                                plt.plot([i, j], [y+h, y+h], lw=1.5, c=col) # i first group, j second group
                                plt.text((i+j)*.5, y+h, label_stat, ha='center', va='bottom', color=col)
                                k+=20 # for plotting
                            elif pval < 0.01:
                                label_stat = "*"
                                y, h, col = data_parameter['Data'].max() + 30 + k, 10, 'k' # determine the height of the annotations
                                plt.plot([i, j], [y+h, y+h], lw=1.5, c=col) # i first group, j second group
                                plt.text((i+j)*.5, y+h, label_stat, ha='center', va='bottom', color=col)
                                k+=20 # for plotting                    
        
            # NON PARAMETRIC test
            else:
                statistics_name = "Kruskal-Wallis"
                statistic, pvalue = kruskal(*data_for_statistics) # pvalue from the kruskal-wallis test
                label_stat = []
                p_value = [] # list of all the p_values from the dunn test
                k = 0
                if pvalue < 0.05: # post-hoc test
                    dunn = sp.posthoc_dunn(data_for_statistics, p_adjust = 'bonferroni') # as Kruskal-Wallis test above, we use a multiple comparison procedure on ranks: Dunn's test. To avoid multiple-testing bias, we perform a bonferroni correction.
                    for i in range (len(dunn)):
                        for j in range (i, len(dunn)): # as repeats in the dataframe, following the diagonal
                            pval = dunn.iat[i,j] # p_value for the comparison between groups i and j; dunn dataframe, ie use .iat
                            p_value.append(f"{pval:.1E}")
                            if 0.01 < pval < 0.05:
                                label_stat = "*"
                                y, h, col = data_parameter['Data'].max() + 30 + k, 10, 'k' # determine the height of the annotations
                                plt.plot([i, j], [y+h, y+h], lw=1.5, c=col) # i first group, j second group
                                plt.text((i+j)*.5, y+h, label_stat, ha='center', va='bottom', color=col)
                                k+=20 # for plotting
                            elif 0.001 < pval < 0.01:
                                label_stat = "**"
                                y, h, col = data_parameter['Data'].max() + 30 + k, 10, 'k' # determine the height of the annotations
                                plt.plot([i, j], [y+h, y+h], lw=1.5, c=col)
                                plt.text((i+j)*.5, y+h, label_stat, ha='center', va='bottom', color=col)
                                k+=20 # for plotting
                            elif pval < 0.01:
                                label_stat = "***"
                                y, h, col = data_parameter['Data'].max() + 30 + k, 10, 'k' # determine the height of the annotations
                                #plt.plot([i, i, j, j], [y, y+h, y+h, y], lw=1.5, c=col)
                                plt.plot([i, j], [y+h, y+h], lw=1.5, c=col)
                                plt.text((i+j)*.5, y+h, label_stat, ha='center', va='bottom', color=col)
                                k+=20 # for plotting
                            
                else:
                    label_stat = "ns (p>0.05)"
        
        # Number of samples, for indication under the plot
        number_samples = []
        for i in data_for_statistics:
            n = len(i)
            number_samples.append(n)
    


    # ================= VIOLIN + SCATTER PLOT ==================   
    
    def violin_plot(self, data_parameter, my_palette, legend, title, group_name, data_for_statistics, check_normality, check_homoscedasticity, mean_list):
        fig, axis = plt.subplots()
        plt.style.use('ggplot')
        sns.violinplot(data=data_parameter, x="Group", y="Data", color = 'lavender', ax=axis)
        sns.swarmplot(data=data_parameter, x="Group", y="Data", palette = my_palette, ax=axis, size = 3, zorder=1)
        # statistical annotation
        self.statistics(group_name, data_for_statistics, data_parameter, check_normality, check_homoscedasticity)
        axis.text(0.5,-0.5, 
                  "The statistical test used is " + statistics_name + '. \n' 
                  + "p_value: " + str(p_value) + '. \n' 
                  + "The number of samples is: " + str(number_samples) + '. \n' 
                  + "The means are: " + str(mean_list),
                  size=12, ha="center", transform=axis.transAxes)
        plt.ylabel(legend)
        plt.suptitle(title)
        plt.subplots_adjust(bottom=0.3)
        plt.margins(0.08) # add some padding around the plot
        plt.show()
        
            
    
    # ================= HISTOGRAM PLOT, for branch distance by branch type ie for 1 file only ==================  
    def branch_type_hist_plot(self, branch_data, parameter, title):
        legend = "Branch distance by branch type"
        fig, axis = plt.subplots()
        branch_data.hist(column='branch-distance', by='branch-type', bins=100, ax=axis)
        axis.set_ylabel(legend)
        plt.suptitle(title)
        plt.show()
     
            
    
    # ================= PLOTTING ==================  
    def button_plot(self):
        global k # used in statistics
        
        parameter = 0
        title = 0
        parameter = self.combobox_var.get()
        title = self.entry_group_0.get()
        
        
        # ============ Check the entries ============ 
        
        if title == 0:
            self.message = messagebox.showinfo(title = "Error", message = "Please add a title")
        if parameter == 0:
            self.message = messagebox.showinfo(title = "Error", message = "Please add a parameter to analyse")
        

        # ============ PLOTTING for the first parameter ============  
        
        if parameter == "Branch distance by branch type - only for 1 file, blue color":
            branch_data = pd.read_excel(files[0][0])
            self.branch_type_hist_plot(branch_data, parameter, title)
            branch_data = None
        
        
        # ============ PLOTTING for the other parameters ============ 
        
        elif parameter == "Number branches" or parameter == "Number isolated branches" or parameter == "Number endpoint-junction branches" or parameter == "Number junction-junction branches" or parameter == "Number circular branches" or parameter == "Number junctions" or parameter == "Number endpoints" or parameter == "Average branch length" or parameter == "Total length" or parameter == "Junctions per volume" or parameter == "Branches per volume" or parameter == "Area" or parameter == "Volume" or parameter == "V percentage occupancy" or parameter == "Diameter_mean_horizontal" or parameter == "Diameter_mean_vertical" or parameter == "Diameter_horizontal" or parameter == "Diameter_vertical":
        
            # Get the groups' colors and names
            color_palette = [self.combobox_color_group1_var.get(), self.combobox_color_group2_var.get(), self.combobox_color_group3_var.get(), self.combobox_color_group4_var.get(), self.combobox_color_group5_var.get(), self.combobox_color_group6_var.get()]
            if 'Color1' in color_palette: # keep only the color names, not the empty fields
                color_palette.remove('Color1')
            if 'Color2' in color_palette:
                color_palette.remove('Color2')
            if 'Color3' in color_palette:
                color_palette.remove('Color3')
            if 'Color4' in color_palette:
                color_palette.remove('Color4')
            if 'Color5' in color_palette:
                color_palette.remove('Color5')
            if 'Color6' in color_palette:
                color_palette.remove('Color6')

                    
            self.group1_name = self.entry_group1.get()
            self.group2_name = self.entry_group2.get()
            self.group3_name = self.entry_group3.get()
            self.group4_name = self.entry_group4.get()
            self.group5_name = self.entry_group5.get()
            self.group6_name = self.entry_group6.get()
            group_name = [self.group1_name, self.group2_name, self.group3_name, self.group4_name, self.group5_name, self.group6_name]
            if '' in group_name: # keep only the input names, not the empty fields
                group_name.remove('')
            if '' in group_name:
                group_name.remove('')
            if '' in group_name:
                group_name.remove('')
            if '' in group_name:
                group_name.remove('')
            if '' in group_name:
                group_name.remove('')
            if '' in group_name:
                group_name.remove('')

            
            if len(color_palette) < len(files): # if the user forgot to input the colors
                while len(color_palette) < len(files):
                    color_palette.append("red")
                    
            if len(group_name) < len(files): # if the user forgot to input the group names
                i = 0
                while len(group_name) < len(files):
                    i+=1
                    group_name.append(("Group " + str(i)))
                
            i=0 # associated to the group n°i
            data_parameter = pd.DataFrame(columns = ['Data', 'Group']) # dataframe from which we'll plot
            my_palette = {} # colors
            dataa = [] # transient list, to add the data of group n°i
            data_for_statistics = [] # final list of the data of each group
            groupp = [] # final list where all the groups' names are listed
            mean_list = [] # list of the means for each group
            
            for group_files in files: # files retrieved from select_files_group_i
                full_data, legend = self.read_excel(group_files, parameter) # full_data = data from the sheets grouped together in a "group"
                k = len(full_data)
                group = group_name[i]
                group_list = [group]*k # name of the group, repeated the nb of times there is a data in full_data - for adding in the dataframe
                color = color_palette[i]
                i+=1
                palette_transient = {str(group): str(color)}
                my_palette.update(palette_transient)
                for k in full_data: # add the data of one group in dataa
                    dataa.append(k)
                for j in group_list:
                    groupp.append(j)
                data_for_statistics.append(full_data)
                means = mean(full_data)
                mean_list.append(f"{means:.1E}")
            data_parameter_transient = {'Data': dataa, 'Group': groupp}
            data_parameter = pd.DataFrame(data_parameter_transient) # final datasheet from which we'll plot
            
            
            # ================ Statistical assumptions ===============================
            
            response = messagebox.askokcancel("Statistical assumptions", "Please check hereafter the statistical assumptions for your data") # 1st parameter Title bar, 2nd one is the message we want to show in the popup
            if response:
                text0, text1, text2, check_normality, check_homoscedasticity = self.check_assumptions(data_for_statistics)
                self.visualize_response = messagebox.showinfo("Statistical assumptions", str(text0) + '\n' + str(text1) + '\n' + str(text2))        


            # ========================= Plotting =====================================
            
            self.violin_plot(data_parameter, my_palette, legend, title, group_name, data_for_statistics, check_normality, check_homoscedasticity, mean_list)
            

        # ============ RENEW the buttons ============  
        
        self.plot_button = customtkinter.CTkButton(master=self.frame_left, text="Plot",
                                                   command = self.button_plot,
                                                   fg_color = "turquoise1")
        self.plot_button.grid(row = 15, column = 0, padx=2, pady=20, sticky="we")

            
        
        



        

        







#===============================APPEARANCE================================================            


    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
        
        if new_appearance_mode == "Light":
            self.button_group1 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 1",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group1)
            self.button_group1.grid(row=10, column=1, pady=10, padx=20, sticky="w")
            # Choose files = group 2
            self.button_group2 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 2",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group2)
            self.button_group2.grid(row=10, column=2, pady=10, padx=20, sticky="w")
            # Choose files = group 3
            self.button_group3 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 3",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group3)
            self.button_group3.grid(row=10, column=3, pady=10, padx=20, sticky="w")
            # Choose files = group 4
            self.button_group4 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 4",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group4)
            self.button_group4.grid(row=10, column=4, pady=10, padx=20, sticky="w")
            # Choose files = group 5
            self.button_group5 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 5",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group5)
            self.button_group5.grid(row=10, column=5, pady=10, padx=20, sticky="w")
            # Choose files = group 6
            self.button_group6 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 6",
                                                         fg_color = "gray49",
                                                         command = self.select_files_group6)
            self.button_group6.grid(row=10, column=6, pady=10, padx=20, sticky="w")
            
            
            
            #[ COLOR boxes]
            self.combobox_color_group1_var = customtkinter.StringVar(value="Color1")
            self.combobox_color_group1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group1_var,
                                                                     fg_color = "gray49")
            self.combobox_color_group1.grid(row=9, column = 0, pady=2, padx=20, sticky="we")

            self.combobox_color_group2_var = customtkinter.StringVar(value="Color2")
            self.combobox_color_group2 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group2_var,
                                                                     fg_color = "gray49")
            self.combobox_color_group2.grid(row = 10, column = 0, pady=2, padx=20, sticky="we")
            
            self.combobox_color_group3_var = customtkinter.StringVar(value="Color3")
            self.combobox_color_group3 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group3_var,
                                                                     fg_color = "gray49")
            self.combobox_color_group3.grid(row = 11, column = 0, pady=2, padx=20, sticky="we")
            
            self.combobox_color_group4_var = customtkinter.StringVar(value="Color4")
            self.combobox_color_group4 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group4_var,
                                                                     fg_color = "gray49")
            self.combobox_color_group4.grid(row = 12, column = 0, pady=2, padx=20, sticky="we")
            
            self.combobox_color_group5_var = customtkinter.StringVar(value="Color5")
            self.combobox_color_group5 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group5_var,
                                                                     fg_color = "gray49")
            self.combobox_color_group5.grid(row = 13, column = 0, pady=2, padx=20, sticky="we")

            self.combobox_color_group6_var = customtkinter.StringVar(value="Color6")
            self.combobox_color_group6 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group6_var,
                                                                     fg_color = "gray49")
            self.combobox_color_group6.grid(row = 14, column = 0, pady=2, padx=20, sticky="we")

            
        if new_appearance_mode == "Dark":
            self.button_group1 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 1 ",
                                                         fg_color = "gray12",
                                                         command = self.select_files_group1)
            self.button_group1.grid(row=10, column=1, pady=10, padx=20, sticky="w")
            # Choose files = group 2
            self.button_group2 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 2",
                                                         fg_color = "gray12",
                                                         command = self.select_files_group2)
            self.button_group2.grid(row=10, column=2, pady=10, padx=20, sticky="w")
            # Choose files = group 3
            self.button_group3 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 3",
                                                         fg_color = "gray12",
                                                         command = self.select_files_group3)
            self.button_group3.grid(row=10, column=3, pady=10, padx=20, sticky="w")
            # Choose files = group 4
            self.button_group4 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 4",
                                                         fg_color = "gray12",
                                                         command = self.select_files_group4)
            self.button_group4.grid(row=10, column=4, pady=10, padx=20, sticky="w")
            # Choose files = group 5
            self.button_group5 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 5",
                                                         fg_color = "gray12",
                                                         command = self.select_files_group5)
            self.button_group5.grid(row=10, column=5, pady=10, padx=20, sticky="w")
            # Choose files = group 6
            self.button_group6 = customtkinter.CTkButton(master = self.frame_right, 
                                                         text = "Group 6",
                                                         fg_color = "gray12",
                                                         command = self.select_files_group6)
            self.button_group6.grid(row=10, column=6, pady=10, padx=20, sticky="w")
            
            
            
            #[COLOR boxes]
            self.combobox_color_group1_var = customtkinter.StringVar(value="Color1")
            self.combobox_color_group1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group1_var,
                                                                     fg_color = "gray12")
            self.combobox_color_group1.grid(row=9, column = 0, pady=2, padx=20, sticky="we")

            self.combobox_color_group2_var = customtkinter.StringVar(value="Color2")
            self.combobox_color_group2 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group2_var,
                                                                     fg_color = "gray12")
            self.combobox_color_group2.grid(row = 10, column = 0, pady=2, padx=20, sticky="we")
            
            self.combobox_color_group3_var = customtkinter.StringVar(value="Color3")
            self.combobox_color_group3 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group3_var,
                                                                     fg_color = "gray12")
            self.combobox_color_group3.grid(row = 11, column = 0, pady=2, padx=20, sticky="we")
            
            self.combobox_color_group4_var = customtkinter.StringVar(value="Color4")
            self.combobox_color_group4 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group4_var,
                                                                     fg_color = "gray12")
            self.combobox_color_group4.grid(row = 12, column = 0, pady=2, padx=20, sticky="we")
            
            self.combobox_color_group5_var = customtkinter.StringVar(value="Color5")
            self.combobox_color_group5 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group5_var,
                                                                     fg_color = "gray12")
            self.combobox_color_group5.grid(row = 13, column = 0, pady=2, padx=20, sticky="we")

            self.combobox_color_group6_var = customtkinter.StringVar(value="Color6")
            self.combobox_color_group6 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                                     values=["blue", "deepskyblue", "royalblue", "navy", "midnightblue", "slateblue", "darkviolet", "aqua", "red", "firebrick", "lightcoral", "crimson", "magenta", "gold", "orange", "green", "lawngreen", "lime", "black", "dimgray", "slategrey", "lightgray"],
                                                                     variable =  self.combobox_color_group6_var,
                                                                     fg_color = "gray12")
            self.combobox_color_group6.grid(row = 14, column = 0, pady=2, padx=20, sticky="we")


    def on_closing(self, event=0):
        self.destroy()




if __name__ == "__main__":
    app = App()
    app.mainloop()
