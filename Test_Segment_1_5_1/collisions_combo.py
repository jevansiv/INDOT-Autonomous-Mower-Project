# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 21:31:04 2023

@author: Evans Lab Laptop
"""

import os
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus.flowables import Flowable
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from gmplot import GoogleMapPlotter
from random import random
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from PIL import Image as Image2
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import imageio
import yaml
import math

class CustomGoogleMapPlotter(GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom, apikey='',
                 map_type='satellite'):
        if apikey == '':
            try:
                with open('apikey.txt', 'r') as apifile:
                    apikey = apifile.readline()
            except FileNotFoundError:
                pass
        super().__init__(center_lat, center_lng, zoom, apikey)

        self.map_type = map_type
        assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])

    def write_map(self,  f):
        f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
                (self.center[0], self.center[1]))
        f.write('\t\tvar myOptions = {\n')
        f.write('\t\t\tzoom: %d,\n' % (self.zoom))
        f.write('\t\t\tcenter: centerlatlng,\n')

        # Change this line to allow different map types
        f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))

        f.write('\t\t};\n')
        f.write(
            '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
        f.write('\n')

    def color_scatter(self, lats, lngs, values=None, colormap='coolwarm',
                      size=2, marker=False, s=None, **kwargs):
        def rgb2hex(rgb):
            """ Convert RGBA or RGB to #RRGGBB """
            rgb = list(rgb[0:3]) # remove alpha if present
            rgb = [int(c * 255) for c in rgb]
            hexcolor = '#%02x%02x%02x' % tuple(rgb)
            return hexcolor

        if values is None:
            #colors = [None for _ in lats]
            colors = ['#FF0000'] * len(lats)
        else:
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=min(values), vmax=max(values))
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
        for lat, lon, c in zip(lats, lngs, colors):
            self.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker,
                         s=s, **kwargs)


from shapely.geometry import Point, Polygon, LinearRing

def calculate_polygon_area(latitude_list, longitude_list):
    # Ensure the polygon is closed
    if latitude_list[0] != latitude_list[-1] or longitude_list[0] != longitude_list[-1]:
        latitude_list.append(latitude_list[0])
        longitude_list.append(longitude_list[0])

    # Check if there are at least 3 coordinates
    if len(latitude_list) < 3 or len(longitude_list) < 3:
        print("Insufficient coordinates for creating a polygon.")
        return 0

    # Create a Shapely Polygon object
    polygon = Polygon(zip(longitude_list, latitude_list))

    # Calculate the area of the polygon using Shapely's area function
    area_square_meters = polygon.area

    # Convert the area to square kilometers (1 square meter = 0.000001 square kilometers)
    area_square_km = area_square_meters * 0.000001

    return area_square_km

from shapely.ops import unary_union

def identify_covered_area(region_latitudes, region_longitudes, latitudes, longitudes):
    # Create a polygon representing the entire region
    region_polygon = Polygon(zip(region_longitudes, region_latitudes))

    # Calculate the area of the entire region
    total_area = calculate_polygon_area(region_latitudes, region_longitudes)

    # Identify and subtract the area covered by the list of locations within the region
    remaining_area = total_area

    if len(latitudes) < 3 or len(longitudes) < 3:
        print("Insufficient coordinates for creating a polygon.")
        return total_area, remaining_area

    # Create a list to store removed polygons
    removed_polygons = []

    for latitude, longitude in zip(latitudes, longitudes):
        point = Point(longitude, latitude)
        if region_polygon.contains(point):
            covered_area = calculate_polygon_area([latitude], [longitude])
            remaining_area -= covered_area

            # Create a new polygon by subtracting the covered area
            covered_polygon = Polygon(zip([longitude], [latitude]))
            region_polygon = region_polygon.difference(covered_polygon)
            
            # Store the removed polygon for visualization or further analysis
            removed_polygons.append(covered_polygon)

    return total_area, remaining_area, removed_polygons

def convert_gps_to_meters(lat1,lat2,lon1,lon2):
    dist_lat = (lat2-lat1)*111139
    dist_long = (lon2-lon1)*111139
    return dist_lat, dist_long

def generate_polygon(latitudes, longitudes):
    """
    lat_vals = []
    long_vals = []
    for i in range(len(latitudes)):
    """


    return Polygon(zip(longitudes, latitudes))

def generate_grid_points(min_x, max_x, min_y, max_y, step_size=0.5):

    grid_points = []
    for x in np.arange(min_x, max_x + step_size, step_size):
        for y in np.arange(min_y, max_y + step_size, step_size):
            grid_points.append((x, y))

    return grid_points

def generate_random_points_in_square(square, num_points):
    min_x, min_y, max_x, max_y = square.bounds
    random_points = []
    for _ in range(num_points):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        random_points.append((x, y))
    return random_points

def filter_points(original_points, filter_points, distance_threshold=0.5):
    index_length = len(original_points)
    covered_points = []
    for i in range(len(filter_points)):
        ii = 0
        while ii < index_length:
            original_x, original_y = original_points[ii]
            filter_x, filter_y =  filter_points[i]
            x_dist = original_x - filter_x
            y_dist = original_y - filter_y
            distance_calc = ((x_dist * x_dist) + (y_dist * y_dist))**(0.5)
            if distance_calc < distance_threshold:
                covered_points.append(filter_points[i])
                ii = index_length
            ii = ii + 1

    return covered_points

def find_rectangle_corners(image):
    # Find the coordinates of zero (black) pixels
    black_pixels = np.argwhere(image == 0)

    if len(black_pixels) == 0:
        raise ValueError("No black pixels found in the image")

    # Calculate the corners of the rectangle
    min_x, min_y = np.min(black_pixels, axis=0)
    max_x, max_y = np.max(black_pixels, axis=0)

    # Calculate the center point
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    return (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (center_x, center_y)

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def identify_origin_resolution(data):
    # Check if the required fields are present
    if 'origin' not in data or 'resolution' not in data:
        raise ValueError("The YAML file does not contain 'origin' and/or 'resolution' fields.")

    origin = data['origin']
    resolution = data['resolution']

    #print("Origin:", origin)
    #print("Resolution:", resolution)
    return origin, resolution

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface
    using the Haversine formula.
    """
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

def determine_side(point, line_start, line_end):
    """
    Determine if a point is on the left or right side of the line formed
    by two other points.
    """
    x1, y1 = line_start
    x2, y2 = line_end
    x, y = point

    # Calculate the cross product
    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    # Points on the left side have a positive cross product, points on the right side have a negative cross product
    if cross_product > 0:
        return "Left"
    elif cross_product < 0:
        return "Right"
    else:
        return "On the line"


def image_selection(csv_title):
    if "oscar" in str(csv_title):
        return "oscar_solo"
    elif "rc" in str(csv_title):
        return "rc_mower_solo"
    else:
        return "tractor_solo"

def sensor_check(csv_file):
    df = pd.read_csv(csv_file)
    
    if df.empty or df.columns.size == 0:
        print("No sensor data")
        return np.zeros(10)
    
    sensor_index_array = np.zeros(10)
    
    for value in df.iloc[:, 0]:
        sensor_name = str(value).strip().lower()
        
        if sensor_name == 'imu':
            sensor_index_array[0] = 1
        elif sensor_name == 'gps':
            sensor_index_array[1] = 1
        elif sensor_name == 'depth':
            sensor_index_array[2] = 1
        elif sensor_name == 'rgb':
            sensor_index_array[3] = 1
        elif sensor_name == 'radar':
            sensor_index_array[4] = 1
        elif sensor_name == 'lidar':
            sensor_index_array[5] = 1
        elif sensor_name == 'flir':
            sensor_index_array[6] = 1
        elif sensor_name == 'ultrasonic':
            sensor_index_array[7] = 1
        elif sensor_name == 'stereo':
            sensor_index_array[8] = 1
        elif sensor_name == 'collision':
            sensor_index_array[9] = 1
    
    return sensor_index_array


print("Do you want simulation results or real-world results?")
print("~Fix this by making different column label on csv of sim - detect its a sim csv or a real-world csv that way~")
print("1 for sim, 2 for real:")
domain_check = int(input())



# Replace 'your_file.csv' with the path to your CSV file
directory_path = 'path_to_project/Test_Segment_1_5_1/'
if directory_path == 'path_to_project/Test_Segment_1_5_1':
    print("Update path at line 326")
directory_path_collision = directory_path + 'CollisionLogs/'
#file_path = directory_path_collision + 'collisions_12_2_2023_0_4.csv'
directory_path_terrain = directory_path + 'TerrainCoverage/'
directory_path_entries = directory_path + 'RoadEntries/'
directory_path_vehicle = directory_path + 'VehicleLog/'
directory_path_sensor = directory_path + 'SensorLogs/'

csv_files = [f for f in os.listdir(directory_path_collision) if f.endswith('.csv')]

if csv_files:
    most_recent_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory_path_collision, x)))
    most_recent_file_path = os.path.join(directory_path_collision, most_recent_file)
else:
    print("No CSV files found in the directory.")
    
    
# Read the CSV file into a DataFrame
if most_recent_file:
    df = pd.read_csv(most_recent_file_path)
    # Perform data processing on the DataFrame 'df' as needed.
else:
    print("No CSV file found to read.")

# Create a PDF for the file
doc = SimpleDocTemplate(directory_path + "collision_summary.pdf", pagesize=landscape(letter))

sign = 0
guardrail = 0
tree = 0
index = np.arange(0,len(df),1)
i = 0
for rows in index:
    df_index = df.iat[i,2]
    i = i + 1
    if df_index == "ParkingSign":
        sign = sign + 1
    if df_index == "GuardRail":
        guardrail = guardrail + 1
    if df_index == "Tree":
        tree = tree + 1

total_collision_count = sign+guardrail+tree

        
csv_files = [f for f in os.listdir(directory_path_terrain) if f.endswith('.csv')]

if csv_files:
    most_recent_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory_path_terrain, x)))
    most_recent_file_path = os.path.join(directory_path_terrain, most_recent_file)
else:
    print("No CSV files found in the directory.")
    
    
# Read the CSV file into a DataFrame
if most_recent_file:
    df_terrain = pd.read_csv(most_recent_file_path)
    # Perform data processing on the DataFrame 'df' as needed.
else:
    print("No CSV file found to read.")
    
dirt = 0
road = 0
grass = 0

index = np.arange(0,len(df_terrain),1)
total_terrain = len(df_terrain)
i = 0
for rows in index:
    df_index = df_terrain.iat[i,0]
    i = i + 1
    if df_index == "Dirt":
        dirt = dirt + 1
    if df_index == "Road":
        road = road + 1
    if df_index == "Grass":
        grass = grass + 1
        
total_on_road = dirt + road
percent_on_road = round((total_on_road/total_terrain) * 100,2)
print("Percent On Road: "+str(percent_on_road)+ " Length:" +str(total_terrain))

# SENSOR DATA
csv_files = [f for f in os.listdir(directory_path_sensor) if f.endswith('.csv')]
if csv_files:
    most_recent_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory_path_sensor, x)))
    most_recent_file_path = os.path.join(directory_path_sensor, most_recent_file)
else:
    print("No CSV files found in the directory.")
# Read the CSV file into a DataFrame
if most_recent_file:
    csv_sensor = most_recent_file_path
else:
    print("No CSV file found to read.")


csv_files = [f for f in os.listdir(directory_path_entries) if f.endswith('.csv')]

if csv_files:
    most_recent_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory_path_entries, x)))
    most_recent_file_path = os.path.join(directory_path_entries, most_recent_file)
else:
    print("No CSV files found in the directory.")
    
    
# Read the CSV file into a DataFrame
if most_recent_file:
    df_entries = pd.read_csv(most_recent_file_path)
    # Perform data processing on the DataFrame 'df' as needed.
else:
    print("No CSV file found to read.")
try:
    road_entries = df_entries['Entries'].iloc[-1]
except:
    road_entries =0

base_lat_deg = # provide base latitutde ##.#######
base_long_deg = # provide base longitude ##.#######

factor = 111139

latitudes = []

for value in df_entries['Latitude']:
    #latitudes.append((float(-value)/(100*factor))+base_lat_deg)
    latitudes.append(float(value))
    

longitudes = []

for value in df_entries['Longitude']:
    #longitudes.append((float(-value)/(100*factor))+base_long_deg)
    longitudes.append(float(value))

    
#road_entries = df_entries.iat[0,0]
        
geo_fence_exits = 3
coverage = 0.87
coverage_percent = coverage * 100




##### MAP CREATION #####



initial_zoom = 20 # was set to 30
num_pts = road_entries




lats = [1.0000000] # provide latitude
lons = [-1.0000000]# provide longitude
print("Update lats/lons if not already done here for origin spot lines 482 and 483")
lats = lats + latitudes
lons = lons + longitudes
values = [0.8]


# Specify the CSV file name
file_name = 'map.csv' #Update map to your csv coodinates
print("Update map coordinates for map.csv, line 489")

# Create the full file path by joining the directory path and file name
file_path = f'{directory_path}/{file_name}'

# Load the CSV file into a DataFrame
df_map = pd.read_csv(file_path)



latitudes_map = []

for value in df_map['Latitude']:
    #latitudes.append((float(-value)/(100*factor))+base_lat_deg)
    latitudes_map.append(float(value))

    

longitudes_map = []

for value in df_map['Longitude']:
    #longitudes.append((float(-value)/(100*factor))+base_long_deg)
    longitudes_map.append(float(value))


# Display the DataFrame

if domain_check == 1:
    print("Simulation Data")
    csv_files = [f for f in os.listdir(directory_path_vehicle) if f.endswith('.csv')]

    if csv_files:
        most_recent_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory_path_vehicle, x)))
        most_recent_file_path = os.path.join(directory_path_vehicle, most_recent_file)
    else:
        print("No CSV files found in the directory.")
        
        
    # Read the CSV file into a DataFrame
    if most_recent_file:
        df_vehicle = pd.read_csv(most_recent_file_path)
        # Perform data processing on the DataFrame 'df' as needed.
    else:
        print("No CSV file found to read.")
        
    latitudes_vehicle = []

    for value in df_vehicle['Latitude']:
        #latitudes.append((float(-value)/(100*factor))+base_lat_deg)
        latitudes_vehicle.append(float(value))
        #print(value)

        
    longitudes_vehicle = []

    for value in df_vehicle['Longitude']:
        #longitudes.append((float(-value)/(100*factor))+base_long_deg)
        longitudes_vehicle.append(float(value))
        #print(value)

else:
    print("Real World Data")
    directory_path_vehicle = '/path/to/directory/'

    if directory_path_vehicle == '/path/to/directory/':
        print("Update path on line 554")
    
    csv_files = [f for f in os.listdir(directory_path_vehicle) if f.endswith('.csv')]

    if csv_files:
        most_recent_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory_path_vehicle, x)))
        most_recent_file_path = os.path.join(directory_path_vehicle, most_recent_file)
    else:
        print("No CSV files found in the directory.")
    
        
    # Read the CSV file into a DataFrame
    if most_recent_file:
        df_vehicle = pd.read_csv(most_recent_file_path)
        # Perform data processing on the DataFrame 'df' as needed.
    else:
        print("No CSV file found to read.")
   
    latitudes_vehicle = []
    #column_index = 1
    
    for value in df_vehicle['Latitude']:
        #latitudes.append((float(-value)/(100*factor))+base_lat_deg)
        latitudes_vehicle.append(float(value))
        #print(value)
    

        
    longitudes_vehicle = []
    #column_index = 2
    
    for value in df_vehicle['Longitude']:
        #longitudes.append((float(-value)/(100*factor))+base_long_deg)
        longitudes_vehicle.append(float(value))
        #print(value)




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
file_path = '/path/to/directory/ros_map_pgm.pgm'

if file_path == '/path/to/directory/ros_map_pgm.pgm':
    print("Update path on line 617")

# Read the PGM file
image = imageio.imread(file_path)

# Assuming the rectangle is black (zero pixel values)
rectangle_corners = find_rectangle_corners(image)

upper_left = rectangle_corners[0]
lower_left = rectangle_corners[1]
lower_right = rectangle_corners[2]
upper_right = rectangle_corners[3]
center_pgm_map = rectangle_corners[4]

#print("Rectangle Corners:", rectangle_corners)
yaml_file_path = '/path/to/directory/ros_map_yaml.yaml'

if yaml_file_path == '/path/to/directory/ros_map_yaml.yaml':
    print("Update path on line 635")


# Read YAML file
yaml_data = read_yaml(yaml_file_path)

# Identify origin and resolution
origin, resolution = identify_origin_resolution(yaml_data)
#print("Origin:", origin)
#print("Resolution:", resolution)
origin_x = -origin[0]/resolution
origin_y = -origin[1]/resolution

upper_left_dist_x = (((upper_left[1] - origin_x)*resolution)/111139) + longitudes_vehicle[0]
upper_left_dist_y = (((upper_left[0] - origin_y)*resolution)/111139) + latitudes_vehicle[0]
lower_left_dist_x = (((lower_left[1] - origin_x)*resolution)/111139) + longitudes_vehicle[0]
lower_left_dist_y = (((lower_left[0] - origin_y)*resolution)/111139) + latitudes_vehicle[0]
upper_right_dist_x = (((upper_right[1] - origin_x)*resolution)/111139) + longitudes_vehicle[0]
upper_right_dist_y = (((upper_right[0] - origin_y)*resolution)/111139) + latitudes_vehicle[0]
lower_right_dist_x = (((lower_right[1] - origin_x)*resolution)/111139) + longitudes_vehicle[0]
lower_right_dist_y = (((lower_right[0] - origin_y)*resolution)/111139) + latitudes_vehicle[0]

x_list_dists = [upper_left_dist_x, lower_left_dist_x, lower_right_dist_x, upper_right_dist_x]
y_list_dists = [upper_left_dist_y, lower_left_dist_y, lower_right_dist_y, upper_right_dist_y]

longitudes_map = []
for value in x_list_dists:
    #latitudes.append((float(-value)/(100*factor))+base_lat_deg)
    longitudes_map.append(float(value))
    #print(value)

latitudes_map = []
for value in y_list_dists:
    #latitudes.append((float(-value)/(100*factor))+base_lat_deg)
    latitudes_map.append(float(value))
    #print(value)



try:
    total_area, remaining_area, removed_polygons = identify_covered_area(latitudes_map, longitudes_map, latitudes_vehicle, longitudes_vehicle)
except:
    total_area = 100.0
    remaining_area = 10.0
    removed_polygons = 1.0

try:
    lat_origin = latitudes_map[1]
    long_origin = longitudes_map[1]

    latitudes_map_meters = []
    longitudes_map_meters = []

    for i in range(len(latitudes_map)):
        lat_converted, long_converted = convert_gps_to_meters(lat_origin,latitudes_map[i],long_origin,longitudes_map[i])
        latitudes_map_meters.append(lat_converted)
        longitudes_map_meters.append(long_converted)

    latitudes_vehicle_meters = []
    longitudes_vehicle_meters = []
    for i in range(len(latitudes_vehicle)):
        lat_converted, long_converted = convert_gps_to_meters(lat_origin,latitudes_vehicle[i],long_origin,longitudes_vehicle[i])
        latitudes_vehicle_meters.append(lat_converted)
        longitudes_vehicle_meters.append(long_converted)
    #polygon_area = generate_polygon(latitudes_map_meters, longitudes_map_meters)
    grid_points = generate_grid_points(min(longitudes_map_meters), max(longitudes_map_meters), min(latitudes_map_meters), max(latitudes_map_meters), step_size = 0.1) # 1 m spacing

    original_points = []
    for i in range(len(latitudes_vehicle_meters)):
        rounded_point = (latitudes_vehicle_meters[i],longitudes_vehicle_meters[i])
        original_points.append(rounded_point)

    covered_points = filter_points(original_points, grid_points, distance_threshold = 5) # 0.33 m buffer
    percentage_for_coverage = ((len(covered_points)/len(grid_points)))*100.0
    print("coverage percentage:" + str(percentage_for_coverage))
except:
    print("Failed to write out area")


print(f"Total area of the region: {total_area} square kilometers")
print(f"Remaining unexplored area inside the region: {remaining_area} square kilometers")

# Print or visualize the removed polygons if needed
print(f"Removed polygons: {removed_polygons}")
#area = calculate_polygon_area(latitudes_map[:-1], longitudes_map[:-1])


import folium
from xyzservices import TileProvider
from statistics import mean

usgs_tiles = TileProvider(
    # Chosen from list at https://leaflet-extras.github.io/leaflet-providers/preview/
    name = "USGS",
    url = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
    attribution = "(C) U.S. Geological Survey",
    html_attribution = "Imagery courtesy of the <a href=\"https://usgs.gov/\">U.S. Geological Survey</a>",
    max_native_zoom = 5,
    max_zoom = 16
)

# max_navtive_zoom = 5, max_zoom = 16
# had testing: max_native_zoom = 20, max_zoom = 19

# Latitude, Longitude
# Example coordinates
upper_real_world_boundary = [37.7749, -122.4194] # San Francisco, CA
lower_real_world_boundary = [34.0522, -118.2437] # Los Angeles, CA


# Calculate distance
distance = haversine(upper_real_world_boundary[0], upper_real_world_boundary[1], lower_real_world_boundary[0], lower_real_world_boundary[1])
print(f"Straight line distance between the two points: {distance:.2f} km")



lat = lats[1:]
longs_ = lons[1:]
try:
    ACRE = folium.Map(location=[mean(lat), mean(longs_)], tiles=usgs_tiles)  # Adjust zoom level
except:
    ACRE = folium.Map(location=[mean(latitudes_vehicle), mean(longitudes_vehicle)], tiles=usgs_tiles)  # Adjust zoom level


track = []



if domain_check == 2:
    ix = 0
    #for i in range(len(latitudes_vehicle))
    # Determine which side of the line the test point is on
    for i in range(len(latitudes_vehicle)):
        test_point = (latitudes_vehicle[i], longitudes_vehicle[i])
        side = determine_side(test_point, (upper_real_world_boundary[0], upper_real_world_boundary[1]), (lower_real_world_boundary[0], lower_real_world_boundary[1]))
        if side == "Right":
            folium.CircleMarker(location=[latitudes_vehicle[i], longitudes_vehicle[i]], radius=0.001, color='orange', fill=True, fill_color='orange', popups='Small CircleMarker').add_to(ACRE)
        else:
            folium.CircleMarker(location=[latitudes_vehicle[i], longitudes_vehicle[i]], radius=0.001, color='red', fill=True, fill_color='red', popups='Small CircleMarker').add_to(ACRE)
            ix = ix + 1
    percent_on_road = round((ix/len(latitudes_vehicle)) * 100,2)
    #print(f"The test point is on the {side} side of the line.")
else:
    for i in range(len(latitudes_vehicle)):
        if df_terrain.iat[i,0] == "Grass":
            folium.CircleMarker(location=[latitudes_vehicle[i], longitudes_vehicle[i]], radius=0.001, color='orange', fill=True, fill_color='orange', popups='Small CircleMarker').add_to(ACRE)
        elif df_terrain.iat[i,0] == "Road" or df_terrain.iat[i,0] == "Dirt":
            folium.CircleMarker(location=[latitudes_vehicle[i], longitudes_vehicle[i]], radius=0.001, color='red', fill=True, fill_color='red', popups='Small CircleMarker').add_to(ACRE)
        


time.sleep(10)
ACRE.fit_bounds(ACRE.get_bounds())
ACRE.save("mapfile.html")




# Set the path to your WebDriver executable (replace with your browser's WebDriver)
#webdriver_path = './chromedriver'

# Load the HTML file with the Google Map
html_file_path = '/path/to/directory/mapfile.html'
if html_file_path == '/path/to/directory/mapfile.html':
    print("Update file path, line 834")

# Create a Selenium WebDriver instance
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run the browser in headless mode (without a visible window)
options.add_argument('--no-sandbox')  # May be needed for headless mode
options.add_argument('--window-size=1920,1080')
# Specify a temporary user data directory (replace 'temp_user_data' with your chosen directory)
options.add_argument('--user-data-dir=/path/to/Test_Segment_1_5_1/temp_user_data')
driver = webdriver.Chrome(options=options)

# Start the WebDriver
driver = webdriver.Chrome(options=options)

# Open the HTML file with the map in the browser
driver.get(f'file:///{html_file_path}')



driver.get('file://' + html_file_path)  # Load the HTML file
time.sleep(5)  # Allow some time for the map to load (adjust as needed)


# Save a screenshot of the map as a PNG image
driver.save_screenshot('mymap_full.png')
#print('Map saved as mymap.png')

# Set the desired dimensions for the screenshot (adjust as needed)
crop_width = 200 #400
crop_height = 150 #300


# Crop the screenshot to the desired area
im = Image2.open('mymap_full.png')
original_width, original_height = im.size

# Calculate the coordinates for the top-left corner of the cropped area
start_x = max(0, (original_width - crop_width) // 2)
start_y = max(0, (original_height - crop_height) // 2)
im_cropped = im.crop((start_x, start_y, start_x + crop_width, start_y + crop_height))
im_cropped.save('mymap.png')
time.sleep(1)





driver.quit()  # Close the browser


# WRITING THE PDF

# Create a list of elements to add to the document
elements = []

# Add the title
title_style = getSampleStyleSheet()["Title"]
title = Paragraph("Safety Report Summary [DRAFT DOCUMENT]", title_style)
elements.append(title)
image_select = image_selection(most_recent_file)
image_select = 'path_to_project/Test_Segment_1_5_1/Logos/'+image_select+'.png'
if image_select == 'path_to_project/Test_Segment_1_5_1/Logos/'+image_select+'.png':
    print("Update path line 915")
image_vehicle = Image(image_select)#, width=300, height=200)
elements.append(image_vehicle)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

# Define a custom paragraph style with center alignment and left indentation
paragraph_style = ParagraphStyle(
    name='ParagraphStyle',
    parent=getSampleStyleSheet()['Normal'],
    fontSize=12,  # You can adjust the font size as needed
    alignment=1,  # Center alignment
    leftIndent=36,  # Left indentation (adjust as needed)
)

paragraph = Paragraph("This report serves as a Simulation Report Card for vehicle performance. The evaluated criterions are: On Road Operations, Coverage, Collisions", paragraph_style)
elements.append(paragraph)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

# Define a custom subtitle style
subtitle_style = ParagraphStyle(
    name='SubtitleStyle',
    parent=getSampleStyleSheet()['Normal'],
    fontSize=14,  # You can adjust the font size as needed
    spaceAfter=12,  # Add space after the subtitle
    alignment=1,  # 0=Left, 1=Center, 2=Right
)

# Add a paragraph entry after the subtitle
paragraph = Paragraph("The On Road percentage quanitifies the percentage of time the vehicle performed on the road. Operations on the road is not allowed.", paragraph_style)
elements.append(paragraph)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space


# Add a subtitle for the table
subtitle = Paragraph("On Road Operations: "+str(percent_on_road)+'%', subtitle_style)
elements.append(subtitle)


elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

# Add a subtitle for the table
subtitle = Paragraph("Road Entries: "+str(road_entries), subtitle_style)
elements.append(subtitle)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

# Add an image
image = Image('mymap.png', width=300, height=200)
elements.append(image)

# Add a paragraph entry after the subtitle

paragraph = Paragraph("The Coverage report quanitifies the total coverage of the desired region to be mowed by the platform.", paragraph_style)
elements.append(paragraph)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

subtitle = Paragraph("Coverage Percent: "+str(round(percentage_for_coverage,2))+"%", subtitle_style)
elements.append(subtitle)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

paragraph = Paragraph("The Mower Collisions Display the total accounts of collisions made with individual types of assets in the Environment.", paragraph_style)
elements.append(paragraph)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

elements.append(Spacer(1, 24))
elements.append(Spacer(1, 24))
elements.append(Spacer(1, 24))
elements.append(Spacer(1, 24))
elements.append(Spacer(1, 24))
elements.append(Spacer(1, 24))

# Add a subtitle for the table
subtitle = Paragraph("Mower Collisions", subtitle_style)
elements.append(subtitle)

data = [
    ["Parking Sign", sign],
    ["Guard Rail", guardrail],
    ["Tree", tree],
]

# Create a table with the labels and integer values
table_data = [["Object", "Collision Count"]] + data
table = Table(table_data, colWidths=[200, 100])
table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
])

table.setStyle(table_style)
elements.append(table)

elements.append(Spacer(1, 24))

subtitle = Paragraph("Onboard Sensors", subtitle_style)
elements.append(subtitle)

sensor_array = sensor_check(csv_sensor)

data_sensor = [["IMU", sensor_array[0]], ["GPS", sensor_array[1]], ["Depth Camera", sensor_array[2]], ["RGB Camera", sensor_array[3]], ["Radar", sensor_array[4]], ["Lidar", sensor_array[5]],
["FLIR", sensor_array[6]], ["Ultrasonic", sensor_array[7]], ["Bump Sensor", sensor_array[9]]]

for sensor in data_sensor:
    sensor[1] = "yes" if sensor[1] == 1 else "-"

table_data = [["Sensor", "Check"]] + data_sensor
checkbox_table = Table(table_data, colWidths=[200, 50])
checkbox_table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
])
checkbox_table.setStyle(table_style)
elements.append(checkbox_table)

# Add an image to the top-left corner

# if you'd like to add an image
#image = Image('/path/to/png.png', width=72, height=72)  # Replace "your_image.png" with the path to your image
#elements.insert(0, image)



# Build the PDF document
doc.build(elements)







