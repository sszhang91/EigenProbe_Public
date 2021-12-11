#!/usr/bin/env python
# coding: utf-8

# In[6]:


from bokeh.io import *
from bokeh.layouts import *
from bokeh.plotting import *
from bokeh.models.renderers import *
from bokeh.palettes import *
from bokeh.models.widgets import *
from bokeh.models import *
from scipy import interpolate
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, Column
import pickle
from pathlib import Path

import os
import copy
import pickle
import numpy as np
from matplotlib import pyplot as plt
from importlib import reload


from scipy.interpolate import splprep, splev

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import warnings
warnings.filterwarnings('ignore')

root = Path(".")

global pt_dict

######################################################################################
#helper method
def save_to_pickle(spline_folder_name,pt_dict,mesh_dict):

    with open(root / spline_folder_name /'pt_dict.pickle', 'wb') as handle:
        pickle.dump(pt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    with open(root / spline_folder_name /'mesh_dict.pickle', 'wb') as handle:
        pickle.dump(mesh_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

def open_pickle(file_name):
    with open(file_name, 'rb') as handle:
        spline_dict = pickle.load(handle)
    return spline_dict

# Sourced from https://gist.github.com/carlodri/66c471498e6b52caf213
def gsf_read(file_name):
    '''Read a Gwyddion Simple Field 1.0 file format
    http://gwyddion.net/documentation/user-guide-en/gsf.html
    
    Args:
        file_name (string): the name of the output (any extension will be replaced)
    Returns:
        metadata (dict): additional metadata to be included in the file
        data (2darray): an arbitrary sized 2D array of arbitrary numeric type
    '''
    if file_name.rpartition('.')[1] == '.':
        file_name = file_name[0:file_name.rfind('.')]
    
    gsfFile = open(file_name + '.gsf', 'rb')
    
    metadata = {}
    
    # check if header is OK
    if not(gsfFile.readline().decode('UTF-8') == 'Gwyddion Simple Field 1.0\n'):
        gsfFile.close()
        raise ValueError('File has wrong header')
        
    term = b'00'
    # read metadata header
    while term != b'\x00':
        line_string = gsfFile.readline().decode('UTF-8')
        metadata[line_string.rpartition(' = ')[0]] = line_string.rpartition('=')[2]
        term = gsfFile.read(1)
        gsfFile.seek(-1, 1)
    
    gsfFile.read(4 - gsfFile.tell() % 4)
    
    #fix known metadata types from .gsf file specs
    #first the mandatory ones...
    metadata['XRes'] = np.int(metadata['XRes'])
    metadata['YRes'] = np.int(metadata['YRes'])
    
    #now check for the optional ones
    if 'XReal' in metadata:
        metadata['XReal'] = np.float(metadata['XReal'])
    
    if 'YReal' in metadata:
        metadata['YReal'] = np.float(metadata['YReal'])
                
    if 'XOffset' in metadata:
        metadata['XOffset'] = np.float(metadata['XOffset'])
    
    if 'YOffset' in metadata:
        metadata['YOffset'] = np.float(metadata['YOffset'])
    
    data = np.frombuffer(gsfFile.read(),dtype='float32').reshape(metadata['YRes'],metadata['XRes'])
    
    gsfFile.close()
    
    return metadata, data
######################################################################################

class tab0():
    
    def __init__(self):

        ### Inputs: Adjusting color code ###
        self.command_mesh4 = Div(text="""<b>Choose color code of plot:</b>""")

        ### Inputs: Get File input ###
        self.file_text = Div(text="""<b>Load image: </b>(accept .gsf (preferred) or .png format)""", width=200, height=20)
        self.file_input = FileInput(accept=".gsf,.png")
        self.color_group_text = Div(text=" If input file format is not .gsf, <b> specify </b> the color range (R,G,B,A) \
                                    for value extraction. Otherwise, ignore.", width=400, height=40)
        self.color_radio_button_group = RadioButtonGroup(labels=['R', 'B','G','A'], active=0,width=300)
        self.previous_spline_text = Div(text="""(Optional) <b>Load previous splines:</b> (only .pickle format)""", width=400, height=20)
        self.spline_file_input = FileInput(accept=".pickle")

class tab1():

    def __init__(self,line_color = '#52fffc'):
        

        self.xrange_text = TextInput(title="X Range max:", value="1",width=100)
        self.yrange_text = TextInput(title="Y Range max:", value="1",width=100)

        self.color_palette_menu = Select(title="Plot color palette:", value='Viridis',options=['Viridis','Cividis','Inferno','Magma','Plasma'],width=150)
        self.cross_color_pick = ColorPicker(color=line_color, title="Cross color:", width=70)
        self.line_color_pick = ColorPicker(color=line_color, title="Line color:", width=70)
        self.line_number_text = TextInput(title="Line index:", value="0",width=100)

        self.color_text = Div(text="""Color bar range:""", width=120, height=20)
        self.color_range_min = TextInput(title="Min:", value="0",width=50)
        self.color_range_max = TextInput(title="Max:", value="1",width=50)

class tab2():

    def __init__(self):

        self.is_boundary_text = Div(text="""Is this domain at the boundary:""", width=200, height=20)
        self.is_boundary_button = RadioButtonGroup(labels=['Yes', 'No'], active=1,width=150)
        self.mirror_text = Div(text="""(If "yes" above) Mirror flip direction:""", width=400, height=20)
        self.mirror_button = RadioButtonGroup(labels=['Left to right', 'Right to left','Up to down','Down to up'], active=0,width=300)

        self.plot_line_button = Button(label="Plot", button_type="primary",width=80)
        self.clear_spline_button = Button(label="Clear previous", button_type="warning",width=100)
        self.save_spline_button = Button(label="Save & Next", button_type="primary",width=80)
        self.save_all_button = Button(label="Save all", button_type="primary",width=150)
        self.save_folder_text = TextInput(title="Save folder name:", value="my_folder",width=300)
        #Mesh
        self.plot_domain_points_button = Button(label="Plot Mesh Coordinates of All Domains", button_type="success", width=50)
        self.plot_circle_button = Button(label="Plot circle", button_type="primary", width = 40)
        self.plot_specific_domain_button = Button(label="Plot Mesh Coordinates of Selected Domains", button_type="success", width=50)
        self.mesh_resolution_input = TextInput(value="30", title="Mesh resolution:", width=50)
        self.smooth_text = Div(text="""<b>Choose to smooth for mesh</b>""", width=200, height=20)
        self.smooth_button = RadioButtonGroup(labels=['Yes', 'No'], active=1,width=150)
        self.optional_text = Div(text="""<b>Note: Performing set oeprations is optional.</b>""", width=300, height=30)
        self.index_text = Div(text="""<b>The merged shape resulting from an intersection or a union take the smallest index of the shapes involved </b>""", width=400, height=30)

       


class tab3():
    def __init__(self):
        self.command_mesh1 = Div(text="""<b>Enter center coordinates for rectangular domain:</b>""")
        self.command_mesh5 = Div(text="""<b>Enter center coordinates for circular domain:</b>""")
        self.command_mesh3 = Div(text="""<b>Enter radius for circular domain:</b>""")
        self.rect_height_input = TextInput(value="", title="height:", width=50)
        self.circle_coordinate_x_input = TextInput(value="0.5", title="x:", width=50)
        self.circle_coordinate_y_input = TextInput(value="0.5", title="y:", width=50)
        self.circle_radius = TextInput(value="", title="r:", width=50)
        self.save_next_circle_rectangle_button = Button(label="Save & Next", button_type="primary", width = 40)
        self.command_mesh2 = Div(text="""<b>Enter parameters for a rectangular domain:</b>""")
        self.rect_width_input = TextInput(value="", title="width:", width=50)
        self.rect_angle_input = TextInput(value="0", title="angle:", width=50)
        self.plot_rect_button = Button(label="Plot rectangle", button_type="primary", width = 40)
        self.fill_button = Button(label="Fill", button_type="primary",width = 40)
        self.run_intersection_button = Button(label="Plot Intersection", button_type = "primary", width = 50)
        self.run_union_button = Button(label="Plot Union", button_type = "primary", width = 50)
        self.run_complement_button = Button(label="Plot Complement", button_type="primary",width = 50)
        self.intersection_input = TextInput(value="0", title="Enter indices of polygons using commas", width = 100)
        self.domain_input_text = TextInput(value="0", title="Enter indices of domains to plot", width = 100)
        self.complement_input_text = TextInput(value="",title="Enter index of domain to complement with (default:1*1 box)")
        self.plot_domain_points_text = Div(text="Will output mesh coordinates for given domain", width=400, height=20)
        self.run_intersection_text = Div(text="""<b>All domains selected must intersect.</b>""", width=400, height=20)
        self.run_union_text = Div(text="""<b>Will return all domains, even if not intersecting. If no overlap, indices will not change.</b>""", width=400, height=20)
        self.run_complement_text = Div(text="""<b>Will return the complement of domains with respect to 1x1 square</b>""", width=400, height=20)
        self.mark_domain_text = Div(text="""<b>Mark points in counter-clockwise order</b>""", width=400, height=20)


class plotter_domain():

    def __init__(self,line_color = '#52fffc',cross_width = 3):

        data = np.random.rand(200,200)
        self.color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=1)
        self.figure1 = figure(x_range=(-0.05*1,1.05*1), y_range=(-0.05*1,1.05*1),plot_width=450, plot_height=400)
        color_bar = ColorBar(color_mapper=self.color_mapper, ticker= BasicTicker(),
                             location=(0,0))
        self.figure1.add_layout(color_bar, 'right')

        self.img = self.figure1.image(image=[data], color_mapper=self.color_mapper,
                           dh=[1.0*1], dw=[1.0*1], x=[0], y=[0])

        self.range_slider = RangeSlider(start=data.min(), end=data.max(),
                                   value=(data.min(), data.max()),
                                   step=0.01, title="range", width=300)


        self.source = ColumnDataSource({'x': [], 'y': []})
        renderer = self.figure1.cross(x='x', y='y', source=self.source, color=line_color, size=10,line_width=cross_width)

        #table
        
        self.line_source = ColumnDataSource({'type': [], '#': []})
        columns = [TableColumn(field="#", title="Line number")]

        self.figure1.toolbar.logo = None
        self.figure1.toolbar.tools = [PanTool(),SaveTool(),UndoTool(),RedoTool(),BoxZoomTool()]

        draw_tool = PointDrawTool(renderers=[renderer])
        cross_hair_tool = CrosshairTool()
        zoom_tool = WheelZoomTool()
        #hover_tool = HoverTool()
        #hover_tool.tooltips = [("index", "@index"),
                    #("(x,y)", "($sx, $sy)")]
        self.figure1.add_tools(draw_tool,cross_hair_tool,zoom_tool)
        self.figure1.toolbar.active_inspect = [cross_hair_tool]
        self.figure1.toolbar.active_tap = draw_tool
        self.figure1.toolbar.active_scroll = zoom_tool  



    







