from bokeh.io import *
from bokeh.layouts import *
from bokeh.plotting import *
from bokeh.models.renderers import *
from bokeh.palettes import *
from bokeh.models.widgets import *
from bokeh.models import *
from PIL import Image
from scipy import interpolate
import scipy.io as sio
import numpy as np
import os
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, Column
from bokeh.layouts import gridplot
from bokeh.events import DoubleTap
import bokeh
from scipy.interpolate import splprep, splev
from sympy import Point, Polygon
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from dolfin import *    

from DomainCanvas import helper
# import helper
import BasovPlasmons.plasmon_modeling as PM
import mshr



#Data used
######################################################################################################

global data, source, line_source, line_folder_name, pt_dict, previous_line, line_color
line_folder_name = "my_folder"
line_color = '#52fffc'
cross_width = 3

class bokeh_object():

    def __init__(self):
        self.pt_dict = {}
        self.domain_coordinates = {}
        self.collection_polygons = {}
        self.mesh_dict = {}

        self.data = None
        self.source = None
        self.line_source = None
        self.line_folder_name = "my_folder"
        self.previous_line = None
        self.line_color = '#52fffc'
        self.plotted_line = False
        self.line_num = 0
        self.computed = 0
        self.complement = 0

    def get_pt_dict(self):
        return self.pt_dict

    def get_collection_polygons(self):
        return self.collection_polygons
    
    def get_domain_coordinates(self):
        return self.domain_coordinates

    def get_mesh_dict(self):
        return self.mesh_dict

    def my_app(self,doc):

        # Widgets instantiation
        ######################################################################################################

        # tab 0 widgets
        title = Div(text="<h1><b>Domain Canvas GUI 1.0</b></h1>",width=600, height=30)
        tab0 = helper.tab0()
        command_mesh4,file_text,file_input,color_group_text,\
            color_radio_button_group,previous_spline_text,spline_file_input = vars(tab0).values()

        space1 = Div(text="""  """, width=200, height=30)
        space2 = Div(text="""  """, width=200, height=20)
        space3 = Div(text="""  """, width=200, height=20)
        space4 = Div(text="""  """, width=200, height=20)
        space5 = Div(text="""  """, width=200, height=20)
        space6 = Div(text="""  """, width=200, height=20)
        space7 = Div(text="""  """, width=40, height=20)
        space8 = Div(text="""  """, width=50, height=20)
        space9 = Div(text="""  """, width=100, height=20)



        # tab 1 widgets
        tab1 = helper.tab1(line_color = line_color)
        xrange_text,yrange_text,color_palette_menu,cross_color_pick,line_color_pick,\
        line_number_text,color_text,color_range_min,color_range_max = vars(tab1).values()

        # tab 2 widgets
        tab2 = helper.tab2()
        is_boundary_text,is_boundary_button,mirror_text,mirror_button,plot_line_button,\
        clear_spline_button,save_spline_button,save_all_button,save_folder_text,plot_domain_points_button,plot_circle_button,plot_specific_domain_button,\
            mesh_resolution_input,smooth_text,smooth_button,optional_text,index_text = vars(tab2).values()

        tab3 = helper.tab3()
        command_mesh1,command_mesh5,command_mesh3,rect_height_input,coordinate_x_input,coordinate_y_input,circle_radius,save_next_circle_rectangle_button,command_mesh2,\
            rect_width_input,rect_angle_input,plot_rect_button,fill_button,run_intersection_button,run_union_button,run_complement_button,intersection_input,domain_input_text,complement_input_text,\
            plot_domain_points_text,run_intersection_text,run_union_text,run_complement_text,mark_domain_text = vars(tab3).values()

        # plot widgets
        global line_source
        plotter_domain = helper.plotter_domain(line_color = line_color,cross_width = cross_width)
        color_mapper,figure1,img,range_slider,source,line_source = vars(plotter_domain).values()


##############################################################################################
        def set_folder_name(attr,old,new):
            global line_folder_name
            line_folder_name = new
        save_folder_text.on_change('value',set_folder_name)

        def save_all(): 
            line_folder_name = save_folder_text.value
            if not os.path.exists(line_folder_name):
                os.makedirs(line_folder_name)

            helper.save_to_pickle(line_folder_name,self.pt_dict,self.mesh_dict)

        save_all_button.on_click(save_all)

        def load_previous(attr,old,new):
            file_name = spline_file_input.filename
            line_dict = helper.open_pickle(file_name)
            global pt_dict
            self.pt_dict.update(line_dict)
            pt_dict.update(line_dict)
            #plot each line
            for index in line_dict:
                x = line_dict[index][0]
                y = line_dict[index][1]
                figure1.line(x, y, line_width=2,color=line_color)
            previous_index = max(int(k) for k, v in line_dict.items())
            line_number_text.value=str(previous_index+1)

        spline_file_input.on_change('value',load_previous)

        def clear_previous_spline():
            source.data['x']=[]
            source.data['y']=[]
            global p
            p.glyph.line_alpha = 0
            xnew = []
            ynew = []    
        clear_spline_button.on_click(clear_previous_spline)

        def save_next_line():
            
            if self.plotted_line == False:
                plot_line()

            x = source.data['x']
            y = source.data['y']
            self.line_num = line_number_text.value
            coordinates = np.stack((x,y))
            self.pt_dict[self.line_num] = coordinates
            
            # reset for next line
            source.data['x']=[]
            source.data['y']=[]
            line_number_text.value=str(int(line_number_text.value)+1)
            is_boundary_button.active = 1
            
        save_spline_button.on_click(save_next_line)

        # Mirror operation
        ######################################################################################################
        def mirror_d2u(xs,ys):
            ymax = float(yrange_text.value)
            ys_mirror = np.flip(-(ys-ymax)+ymax)
            xs_mirror = np.flip(xs)
            x_total = np.append(xs,xs_mirror)
            y_total = np.append(ys,ys_mirror)
            return x_total,y_total

        def mirror_u2d(xs,ys):
            ymax = float(yrange_text.value)
            ys_mirror = np.flip(-(ys+ymax)-ymax)
            xs_mirror = np.flip(xs)
            x_total = np.append(xs,xs_mirror)
            y_total = np.append(ys,ys_mirror)
            return x_total,y_total

        def mirror_l2r(xs,ys):
            xmax = float(xrange_text.value)
            xs_mirror = np.flip(-(xs-xmax)+xmax)
            ys_mirror = np.flip(ys)
            x_total = np.append(xs,xs_mirror)
            y_total = np.append(ys,ys_mirror)
            return x_total,y_total

        def mirror_r2l(xs,ys):
            xmax = float(xrange_text.value)
            xs_mirror = np.flip(-(xs+xmax)-xmax)
            ys_mirror = np.flip(ys)
            x_total = np.append(xs,xs_mirror)
            y_total = np.append(ys,ys_mirror)
            return x_total,y_total

        # Plotting interactions
        ######################################################################################################
        range_callback = CustomJS(args=dict(img=img), code='''
                                img.glyph.color_mapper.low = cb_obj.value[0];
                                img.glyph.color_mapper.high = cb_obj.value[1];
                                ''')
        range_slider.js_on_change('value',range_callback)

        def update_figure(attr, old, new):
            img.glyph.global_alpha = 0

            file_name = file_input.filename
            if file_name[-3:] == 'gsf':
                metadata,data = helper.gsf_read(file_name)
                data = np.flipud(data)
                xrange_text.value = str(metadata['XReal']*1e9)
                yrange_text.value = str(metadata['YReal']*1e9)

                figure1.xaxis.axis_label = 'x'
                figure1.yaxis.axis_label = 'y'

                figure1.image(image=[data], color_mapper=color_mapper,
                               dh=[metadata['YReal']*1e9], dw=[metadata['XReal']*1e9], 
                               x=[-metadata['XReal']*1e9/2], y=[-metadata['YReal']*1e9/2])

                range_slider.start = data.min()
                range_slider.end = data.max()
                range_slider.value = data.min(),data.max()
                renderer = figure1.cross(x='x', y='y', source=source, color=line_color, size=10,line_width=cross_width)
            else:    # presumbly png or jpeg
                im = Image.open(file_name, 'r')
                pix_val = np.array(im.getdata())
                total_length = len(pix_val)
                N = int(np.sqrt(total_length))
                color_index = int(color_radio_button_group.active)
                Lx,Ly = im.size
                data = np.reshape(pix_val[:,color_index],(Ly,Lx))

                data = np.flipud(data)
                xrange_text.value = str(1)
                yrange_text.value = str(1)

                figure1.xaxis.axis_label = 'x'
                figure1.yaxis.axis_label = 'y'

                figure1.x_range.update(start=-0.55, end=0.55)
                figure1.y_range.update(start=-0.55, end=0.55)

                #img1 = 
                figure1.image(image=[data], color_mapper=color_mapper,\
                               dh=[1], dw=[1],x=[-0.5], y=[-0.5])
                range_slider.start = data.min()
                range_slider.end = data.max()
                range_slider.value = data.min(),data.max()
                renderer = figure1.cross(x='x', y='y', source=source, color=line_color, size=10,line_width=cross_width)

            
        file_input.on_change('value',update_figure)
        
        def update_figure_after_color_change():
            img1.glyph.global_alpha = 0

            file_name = file_input.filename
            if file_name[-3:] == 'gsf':

                metadata,data = helper.gsf_read(file_name)
                data = np.flipud(data)
                xrange_text.value = str(metadata['XReal']*1e9)
                yrange_text.value = str(metadata['YReal']*1e9)

                figure1.xaxis.axis_label = 'x'
                figure1.yaxis.axis_label = 'y'

                img2 = figure1.image(image=[data], color_mapper=color_mapper,
                               dh=[metadata['YReal']*1e9], dw=[metadata['XReal']*1e9], 
                               x=[-metadata['XReal']*1e9/2], y=[-metadata['YReal']*1e9/2])
                range_slider.start = data.min()
                range_slider.end = data.max()
                range_slider.value = data.min(),data.max()
                renderer = figure1.cross(x='x', y='y', source=source, color=line_color, size=10,line_width=cross_width)
            elif file_name[-3:] == 'png' or file_name[-3:] == 'jpeg':
                im = Image.open(file_name, 'r')
                pix_val = np.array(im.getdata())
                total_length = len(pix_val)
                N = int(np.sqrt(total_length))
                color_index = int(color_radio_button_group.active)
                Lx,Ly = im.size
                data = np.reshape(pix_val[:,color_index],(Ly,Lx))

                data = np.flipud(data)
                xrange_text.value = str(1)
                yrange_text.value = str(1)

                figure1.xaxis.axis_label = 'x'
                figure1.yaxis.axis_label = 'y'

                figure1.x_range.update(start=-0.55, end=0.55)
                figure1.y_range.update(start=-0.55, end=0.55)

                img2 = figure1.image(image=[data], color_mapper=color_mapper,\
                               dh=[1], dw=[1],x=[-0.5], y=[-0.5])
                range_slider.start = data.min()
                range_slider.end = data.max()
                range_slider.value = data.min(),data.max()
                renderer = figure1.cross(x='x', y='y', source=source, color=line_color, size=10,line_width=cross_width)
            else:
                1 # do nothing

        color_radio_button_group.on_click(update_figure_after_color_change)

        #change plot range
        callback_x = CustomJS(args=dict(x_range=figure1.x_range), code="""
            x_range.setv({"start": -cb_obj.value*0.55, "end": cb_obj.value*0.55})
             """)
        callback_y = CustomJS(args=dict(y_range=figure1.y_range), code="""
            y_range.setv({"start": -cb_obj.value*0.55, "end": cb_obj.value*0.55})
            """)
        xrange_text.js_on_change('value',callback_x)
        yrange_text.js_on_change('value',callback_y)

        def update_color_palette(attr,old,new):
            color_mapper.palette = new+'256'
        color_palette_menu.on_change('value',update_color_palette)

        def update_color_min(attr, old, new):
            range_slider.start = float(new)
        color_range_min.on_change('value',update_color_min)

        def update_color_max(attr, old, new):
            range_slider.end = float(new)
        color_range_max.on_change('value',update_color_max)

        def update_line_color(attr,old,new):
            global line_color
            line_color = new
            p.glyph.line_color=new
            renderer = figure1.cross(x='x', y='y', source=source, color=new, size=10,line_width=cross_width)
        line_color_pick.on_change('color',update_line_color)

        def plot_line():

            x = source.data['x']
            y = source.data['y']
            x = np.array(x)
            y = np.array(y)
            if is_boundary_button.active == 0:   # is boundary

                if mirror_button.active == 0:      #left to right
                    x,y = mirror_l2r(x,y)
                elif mirror_button.active == 1:    #right to left
                    x,y = mirror_r2l(x,y)
                elif mirror_button.active == 2:    #up to down
                    x,y = mirror_u2d(x,y)
                else:                              #down to up
                    x,y = mirror_d2u(x,y)
            x = list(x)
            y = list(y)
            x.append(x[0])
            y.append(y[0])
            source.data['x'] = x
            source.data['y'] = y
            global p
            p = figure1.line(x, y, line_width=2,color=line_color)
            p = figure1.patch(x,y,alpha=0.17,line_width = 2) #shade in domains
            self.plotted_line = True
        plot_line_button.on_click(plot_line)

        #hovertool
        TOOLTIPS = [
                #("index", "@index"),
                ("(x,y)", "($x, $y)"),
            ]
        for key, value in self.pt_dict.items():
            k = int(key)
            source = ColumnDataSource(data=dict(
                x=value[0],
                y=value[1],
                index = k*(1+np.zeros(len(value[0]))),
            ))
        hover = HoverTool(tooltips=[
                ("index", "@index"),
                ("(x,y)", "($x, $y)"),
            ])
        figure1.add_tools(hover)
        figure2 = figure(plot_width = 400, plot_height = 400)

        def run_intersection():
            global p
            self.domain_coordinates = {} #while reviewing, I kind of realized it's unnecessary for this to be a class variable, bc it's only used in plotting.
            if self.computed==0: #self.computed keeps track of how many times intersection, union were used (So far the exact number is not necessary)
                    #the reason to keep track whether  these buttons are pressed or not is because
                    #simply converting self.pt_dict to self.collection_polygons would be inaccurate due to indexing. However, if those buttons haven't been pressed, then we
                    #need to convert self.pt_dict ot collection_polygons. Hence, this if loop.
                for key,values in self.pt_dict.items():
                    coordinates_x = values[0]
                    coordinates_y = values[1]
                    coordinates_tuple = list(zip(coordinates_x,coordinates_y))
                    self.collection_polygons[key] = Polygon(coordinates_tuple)
            if self.computed !=0:
                p.glyph.line_alpha = 0
            indices = intersection_input.value #indices of polygons to intersect
            chunks = indices.split(',')
            list_key_del = [] #a list of indices to delete due to the intersection function. 
            if len(chunks) > 1: # if there are more than one index text input.
                for index1 in chunks: # the complicated for, if loops below compare two polygons in the chunks list and returns the intersection if they intersect, also changing the index.
                    #if they do not intersect, then nothing changes. 
                    for index2 in chunks:
                        poly1 = self.collection_polygons[index1]
                        poly2 = self.collection_polygons[index2]
                        if poly1 != poly2:
                            if poly1.intersects(poly2):
                                intersection = poly1.intersection(poly2)
                                k1=int(index1)
                                k2=int(index2)
                                key_max = str(max(k1,k2))
                                key_min = str(min(k1,k2))
                                self.collection_polygons[key_min] = intersection
                                list_key_del.append(key_max) # the intersection takes the minimum index of the shapes involved. For the revised version, appending at the end, look at union.
                            else:
                                self.collection_polygons = {}
                        else: 
                            continue
                for del_index in list_key_del: #process of destroying unneeded polygons
                    #self.collection_polygons = {key:val for key, val in self.collection_polygons.items() if key != del_index}
                    chunks = [list_key for list_key in chunks if list_key != del_index]
                for index in chunks: 
                    polygon = self.collection_polygons[index]
                    coordinates_array_intersection = np.asarray(polygon.exterior.coords)
                    self.domain_coordinates = coordinates_array_intersection.T
                    p = figure1.line(self.domain_coordinates[0],self.domain_coordinates[1], line_width = 2, color = "red") #convert Polygon to numpy array to draw
            else: #case for 1shape (the .intersects() function above does not work for 1 shape obviously)
                self.domain_coordinates = self.pt_dict['0']
                p = figure1.line(self.domain_coordinates[0],self.domain_coordinates[1], line_width = 2, color = "red")
            self.computed+=1    
        run_intersection_button.on_click(run_intersection)

        def run_union(): #everything is identical to intersection except for line 467, which uses .union. The index revision is also made in lines 474~479
            global p
            if self.computed==0:
                for key,values in self.pt_dict.items():
                    coordinates_x = values[0]
                    coordinates_y = values[1]
                    coordinates_tuple = list(zip(coordinates_x,coordinates_y))
                    self.collection_polygons[key] = Polygon(coordinates_tuple)
            if self.computed !=0:
                p.glyph.line_alpha = 0
            indices = intersection_input.value
            chunks = indices.split(',')
            list_key_del = []
            if len(chunks) > 1:
                for index1 in chunks:
                    for index2 in chunks:
                        poly1 = self.collection_polygons[index1]
                        poly2 = self.collection_polygons[index2]
                        if poly1 != poly2:
                            if poly1.intersects(poly2):
                                union = poly1.union(poly2)
                                k1=int(index1)
                                k2=int(index2)
                                key_max = str(max(k1,k2))
                                key_min = str(min(k1,k2))
                                self.collection_polygons[key_min] = union
                                list_key_del.append(key_max)
                                #length = len(self.collection_polygons)
                                #self.collection_polygons[length] = union
                                #chunks.append(len(self.collection_polygons))
                                #list_key_del.append(index1)
                                #list_key_del.append(index2)
                                #indexing issue: stable version takes the minimum index of shapes involved; we want to append to end
                                
                            else:
                                continue
                        else: 
                            continue
                for del_index in list_key_del:
                    #self.collection_polygons = {key:val for key, val in self.collection_polygons.items() if key != del_index}
                    chunks = [list_key for list_key in chunks if list_key != del_index]
                for index in chunks:
                    polygon = self.collection_polygons[index]
                    coordinates_array_union = np.asarray(polygon.exterior.coords)
                    self.domain_coordinates = coordinates_array_union.T
                    p = figure1.line(self.domain_coordinates[0],self.domain_coordinates[1], line_width = 2, color = "red")
            else: 
                self.domain_coordinates = self.pt_dict['0']
                p = figure1.line(self.domain_coordinates[0],self.domain_coordinates[1], line_width = 2, color = "red")
            self.computed+=1    
        run_union_button.on_click(run_union)
        
        def run_complement(): #up to line 542 everything is the same as union because we need to know all shapes inside in order to calculate complement. If you make the index change, you should update it here as well. 
            global p
            if self.computed==0:
                for key,values in self.pt_dict.items():
                    coordinates_x = values[0]
                    coordinates_y = values[1]
                    coordinates_tuple = list(zip(coordinates_x,coordinates_y))
                    self.collection_polygons[key] = Polygon(coordinates_tuple)
            if self.computed !=0:
                p.glyph.line_alpha = 0
            indices = intersection_input.value
            chunks = indices.split(',')
            list_key_del = []
            if len(chunks) > 1:
                for index1 in chunks:
                    for index2 in chunks:
                        poly1 = self.collection_polygons[index1]
                        poly2 = self.collection_polygons[index2]
                        if poly1 != poly2:
                            if poly1.intersects(poly2):
                                union = poly1.union(poly2)
                                k1=int(index1)
                                k2=int(index2)
                                key_max = str(max(k1,k2))
                                key_min = str(min(k1,k2))
                                self.collection_polygons[key_min] = union
                                list_key_del.append(key_max)
                            else:
                                continue
                        else: 
                            continue
                for del_index in list_key_del:
                    self.collection_polygons = {key:val for key, val in self.collection_polygons.items() if key != del_index}
                    chunks = [list_key for list_key in chunks if list_key != del_index]
                for index in chunks:
                    polygon = self.collection_polygons[index]
                    coordinates_array_union = np.asarray(polygon.exterior.coords)
                    self.domain_coordinates = coordinates_array_union.T
                    p = figure1.line(self.domain_coordinates[0],self.domain_coordinates[1], line_width = 2, color = "red")
            #Error: Have to draw inside shape first
            else: 
                self.domain_coordinates = self.pt_dict['0']
                p = figure1.line(self.domain_coordinates[0],self.domain_coordinates[1], line_width = 2, color = "red")
            self.computed+=1   
            if complement_input_text.value =='': #index input of shape to complement out of. If nothing is inputted, then 1*1box is colored red.
                x_square = [0,0,1,1,0]
                y_square = [1,0,0,1,1]
                figure1.line(x_square,y_square, line_width = 2, color = "red", alpha = 1)
            else: 
                complement_shape = self.pt_dict[complement_input_text.value]
                x = complement_shape[0]
                y = complement_shape[1]
                figure1.line(x,y,line_width = 2, color = "red", alpha = 1)
            self.complement+=1
        run_complement_button.on_click(run_complement)

        def run(): #plot all domains button
            index_str='0'
            mesh_res=30    # the most important parameter: resolution of mesh
            store=False

            # these are tuning parameters that only need to be changed if the kernel dies
            s_param=0
            ulen=100
            k=3
            if self.complement !=0:   # I put this in because the complement button wasn't working. If it works you can just remove this if-else statement. 
                self.pt_dict = 0
            else:
                if self.computed==0: 
                    for key,values in self.pt_dict.items():
                        coordinates_x = values[0]
                        coordinates_y = values[1]
                        coordinates_tuple = list(zip(coordinates_x,coordinates_y))
                        self.collection_polygons[key] = Polygon(coordinates_tuple)
                for key,polygon in self.collection_polygons.items():
                    polygon_oriented = orient(polygon,sign=1.0) #sign=1.0 is counterclockwise for fenics purposes
                    coordinates_array_oriented = np.asarray(polygon_oriented.exterior.coords) #extracting coordinates of Polygon object, not sure how to do this with an object with a hole
                    if smooth_button.active == 0:   # yes smoothing               
                        tck, u = splprep(coordinates_array_oriented.T, u=None, s=s_param, per=1,k=k) 
                        u_new = np.linspace(u.min(), u.max(), ulen)
                        x_new, y_new = splev(u_new, tck, der=0)
                    else:
                        x_new = coordinates_array_oriented.T[0]
                        y_new = coordinates_array_oriented.T[1]
                    pts_new=list(zip(x_new,y_new))
                    Pts=[PM.Point(pair) for pair in pts_new]
                    if self.complement==0:
                        geometry=mshr.Polygon(Pts)
                        mesh = mshr.generate_mesh(geometry, mesh_res, "cgal")
                    else: 
                        geometry=mshr.Rectangle(Point(1,1), Point(0,0))-mshr.Polygon(Pts)
                        mesh = mshr.generate_mesh(geometry, mesh_res, "cgal")
                    self.mesh_dict[key] = mesh #store into mesh dictionary
                    coords = mesh.coordinates()
                    x_points,y_points = mesh.coordinates().T
                    figure2.circle(x_points, y_points, size = 3, color="black")
        plot_domain_points_button.on_click(run)

        def run_select(): #plots meshes of selected domains
            index_str='0'
            mesh_res = float(mesh_resolution_input.value)
            store=False

            # these are tuning parameters that only need to be changed if the kernel dies
            s_param=0
            ulen=100
            k=3
            if self.computed==0: #likewise converts self.pt_dict to Polygon dictionary if union, intersection haven't been pressed.
                for key,values in self.pt_dict.items():
                    coordinates_x = values[0]
                    coordinates_y = values[1]
                    coordinates_tuple = list(zip(coordinates_x,coordinates_y))
                    self.collection_polygons[key] = Polygon(coordinates_tuple)
            indices = domain_input_text.value
            complement_index = complement_input_text.value
            chunks = indices.split(',') #list of indices of domains that are selected
            for index in chunks: 
                #if self.complement == 0:
                polygon = self.collection_polygons[index]
                #else:
                #    polygon = self.collection_polygons['0']
                polygon_oriented = orient(polygon,sign=1.0)
                coordinates_array_oriented = np.asarray(polygon_oriented.exterior.coords)
                if smooth_button.active == 0:   # yes smoothing               
                    tck, u = splprep(coordinates_array_oriented.T, u=None, s=s_param, per=1,k=k) 
                    u_new = np.linspace(u.min(), u.max(), ulen)
                    x_new, y_new = splev(u_new, tck, der=0)
                else:
                    x_new = coordinates_array_oriented.T[0]
                    y_new = coordinates_array_oriented.T[1]
                pts_new=list(zip(x_new,y_new))
                Pts=[PM.Point(pair) for pair in pts_new]
                # reference line 663-- tells whether complement button was pressed or not
                if self.complement==0:
                    geometry=mshr.Polygon(Pts)
                    mesh = mshr.generate_mesh(geometry, mesh_res, "cgal")
                else: 
                    if complement_index == '':
                        geometry=mshr.Polygon([Point(1,0), Point(1,1), Point(0,1), Point(0,0)])- mshr.Polygon(Pts) #null object?
                        mesh = mshr.generate_mesh(geometry, mesh_res, "cgal")
                    else: 
                        complement = self.pt_dict[complement_index]
                        coordinates_x = complement[0]
                        coordinates_y = complement[1]
                        #coordinates_tuple = list(zip(coordinates_x,coordinates_y))
                        pts_new=list(zip(coordinates_x,coordinates_y))
                        outside_pts=[PM.Point(pair) for pair in pts_new] 
                        inner_pts = Pts                       
                        geometry=mshr.Polygon(outside_pts)-mshr.Polygon(inner_pts)
                        mesh = mshr.generate_mesh(geometry, mesh_res, "cgal")
                self.mesh_dict[index] = mesh
                coords = mesh.coordinates()
                x_points,y_points = mesh.coordinates().T
                figure2.circle(x_points, y_points, size = 3, color="black")
        plot_specific_domain_button.on_click(run_select)

        def plot_circle():
            self.line_num = line_number_text.value
            x_input, y_input, r = float(coordinate_x_input.value), float(coordinate_y_input.value), float(circle_radius.value)          
            x=[]
            y=[]
            for i in range (20): #plots 20 points for the circle regardless of radius. Maybe this is a little crude?
                figure1.cross(x_input+r*np.cos(i*np.pi/10), y_input+r*np.sin(i*np.pi/10), size= 7, line_width=cross_width, color=line_color, alpha=1)
                x.append(x_input+r*np.cos(i*np.pi/10))
                y.append(y_input+r*np.sin(i*np.pi/10))
            x.append(x_input+r)
            y.append(y_input)
            coordinates = np.stack((x,y))
            self.pt_dict[self.line_num] = coordinates
            figure1.line(x,y,line_width=2,color=line_color,alpha=1)
            p = figure1.patch(x,y,alpha=0.2,line_width = 2)

        plot_circle_button.on_click(plot_circle)

        def save_next_circle_rectangle(): #line index increment.
            self.line_num = line_number_text.value
            line_number_text.value=str(int(line_number_text.value)+1)   
        save_next_circle_rectangle_button.on_click(save_next_circle_rectangle)  

        def plot_rect(): #complicated because of the angle parameter
            x_input, y_input, w , h, theta = float(coordinate_x_input.value), float(coordinate_y_input.value),\
                float(rect_width_input.value), float(rect_height_input.value), float(rect_angle_input.value)*np.pi/180
            figure1.cross(-w/2*np.cos(theta)-h/2*np.sin(theta)+x_input, -w/2*np.sin(theta)+h/2*np.cos(theta)+y_input, size = 7, line_width=cross_width, color = line_color, alpha = 1)
            figure1.cross(w/2*np.cos(theta)-h/2*np.sin(theta)+x_input, w/2*np.sin(theta)+h/2*np.cos(theta)+y_input, size = 7, line_width=cross_width, color = line_color, alpha = 1)
            figure1.cross(w/2*np.cos(theta)+h/2*np.sin(theta)+x_input, w/2*np.sin(theta)-h/2*np.cos(theta)+y_input, size = 7, line_width=cross_width, color = line_color, alpha = 1)
            figure1.cross(-w/2*np.cos(theta)+h/2*np.sin(theta)+x_input, -w/2*np.sin(theta)-h/2*np.cos(theta)+y_input, size = 7, line_width=cross_width, color = line_color, alpha = 1)

            x = [-w/2*np.cos(theta)-h/2*np.sin(theta)+x_input,-w/2*np.cos(theta)+h/2*np.sin(theta)+x_input,w/2*np.cos(theta)+h/2*np.sin(theta)+x_input,\
                w/2*np.cos(theta)-h/2*np.sin(theta)+x_input]
            y = [-w/2*np.sin(theta)+h/2*np.cos(theta)+y_input,-w/2*np.sin(theta)-h/2*np.cos(theta)+y_input,w/2*np.sin(theta)-h/2*np.cos(theta)+y_input,\
                w/2*np.sin(theta)+h/2*np.cos(theta)+y_input]
            #add first coordinate to draw closed shape
            x_line = [-w/2*np.cos(theta)-h/2*np.sin(theta)+x_input,w/2*np.cos(theta)-h/2*np.sin(theta)+x_input,w/2*np.cos(theta)+h/2*np.sin(theta)+x_input,\
                -w/2*np.cos(theta)+h/2*np.sin(theta)+x_input,-w/2*np.cos(theta)-h/2*np.sin(theta)+x_input]
            y_line = [-w/2*np.sin(theta)+h/2*np.cos(theta)+y_input,w/2*np.sin(theta)+h/2*np.cos(theta)+y_input,w/2*np.sin(theta)-h/2*np.cos(theta)+y_input,\
                -w/2*np.sin(theta)-h/2*np.cos(theta)+y_input,-w/2*np.sin(theta)+h/2*np.cos(theta)+y_input]
            figure1.line(x_line,y_line,line_width=2,color=line_color,alpha=1)
            p = figure1.patch(x,y,alpha=0.2,line_width = 2)
            coordinates = np.stack((x,y))
            self.line_num = line_number_text.value
            self.pt_dict[self.line_num] = coordinates

        plot_rect_button.on_click(plot_rect)
        #I wanted fill to be able to do the shading of domains for complement as well. look at bokeh's multi_polygons
        #def fill():
        #    pt_dict_x=[]
        #    pt_dict_y=[]
        #    a = list(self.pt_dict.values())
        #    for i in range(len(self.pt_dict)):
        #        pt_dict_x.append(a[i][0])
        #        pt_dict_y.append(a[i][1])
        #    figure1.multi_polygons(xs=[[[ pt_dict_x ]]], ys=[[[ pt_dict_y ]]])
        #fill_button.on_click(fill)


        # Layout
        ######################################################################################################
        load = column(
                    command_mesh4,
                    color_group_text,
                    color_radio_button_group,
                    space5,
                    row(file_text,file_input),
                    space6,
                    previous_spline_text,
                    spline_file_input
                    )
        tab0_layout=Panel(child=load,title='Load files')

        plot = column(
                    row(xrange_text,yrange_text),
                    space2,
                    row(color_palette_menu,line_color_pick),
                    row(color_text,color_range_min,color_range_max),
                    range_slider,
                    )
        tab1_layout = Panel(child=plot,title='Adjust plot')

        mark = column(
                    mark_domain_text,
                    is_boundary_text,
                    is_boundary_button,
                    mirror_text,
                    mirror_button,
                    space3,
                    row(plot_line_button,clear_spline_button,save_spline_button)
                    )
        tab_mark_layout = Panel(child=mark,title='Polygon Domain')

        circle = column(
                    command_mesh5,
                    row(coordinate_x_input,coordinate_y_input),
                    command_mesh3,
                    circle_radius,
                    row(plot_circle_button, space7, save_next_circle_rectangle_button, space8),
                    space1
        )
        tab_circle_layout = Panel(child=circle, title='Circular Domain')
                    
        rectangle = column(            
                    command_mesh1,
                    row(coordinate_x_input,coordinate_y_input),
                    command_mesh2,
                    row(rect_height_input,rect_width_input,rect_angle_input),
                    row(plot_rect_button, space8, save_next_circle_rectangle_button, space8)
        )
        tab_rect_layout = Panel(child=rectangle, title='Rectangular Domain')

        save_file = column(
                    save_folder_text,
                    save_all_button
        )
        tab_save_layout = Panel(child=save_file, title='Save File')

        specify_domain = column(
                    line_number_text,
                    Tabs(tabs=[tab_mark_layout,tab_circle_layout,tab_rect_layout],active = 0)
        )
        tab_specify_domain_layout = Panel(child = specify_domain, title = 'Specify Domain')

        intersection = column(
                    run_intersection_text,
                    intersection_input,
                    run_intersection_button 
        )
        tab_intersection_layout = Panel(child=intersection, title='Intersection')

        union = column(
                    run_union_text,
                    space8,
                    intersection_input,
                    space9,
                    run_union_button
        )
        tab_union_layout = Panel(child=union, title='Union')

        complement = column(
                    run_complement_text,
                    space9,
                    intersection_input,
                    run_complement_button,
                    complement_input_text
        )
        tab_complement_layout = Panel(child=complement, title='Complement')

        special_functions = column(
                    optional_text,
                    space9,
                    index_text,
                    Tabs(tabs=[tab_intersection_layout,tab_union_layout,tab_complement_layout])
        )
        tab_special_functions_layout = Panel(child=special_functions,title='Set Operations')

        plotting_mesh = column(
                    mesh_resolution_input,
                    smooth_text,
                    smooth_button,
                    domain_input_text,
                    plot_specific_domain_button,
                    plot_domain_points_button,
                    save_folder_text,
                    save_all_button
        )
        tab_plotting_mesh_layout = Panel(child=plotting_mesh,title='Mesh')

        plots= row(figure1,figure2)
        tabs_plotting = Panel(child=plots,title='Plots')

        tabs = Tabs(tabs=[tab0_layout,tab1_layout,tab_specify_domain_layout,tab_special_functions_layout,tab_plotting_mesh_layout],active=0,width=420)

        all_layout = column(title,space1,space4,row(tabs,plots))


        doc.add_root(all_layout)
        doc.title = "Domain Canvas GUI"