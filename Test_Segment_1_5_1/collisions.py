r# -*- coding: utf-8 -*-
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


# Replace 'your_file.csv' with the path to your CSV file

directory_path = '/path/to/directory/'

if directory_path == '/path/to/directory/':
    print("Update path on line 22")

directory_path_collison = directory_path + 'CollisionLogs/'
file_path = directory_path_collisions + 'collisions_12_2_2023_0_31.csv'
directory_path_terrain = directory_path + 'TerrainCoverage/'
directory_path_entries = directory_path + 'RoadEntries/'

csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

if csv_files:
    most_recent_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory_path, x)))
    most_recent_file_path = os.path.join(directory_path, most_recent_file)
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
    if df_index == "Grass":
        road = road + 1
    if df_index == "Grass":
        grass = grass + 1
        
total_on_road = dirt + road
percent_on_road = round((dirt/total_terrain) * 100,2)

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

road_entries = df_entries.iat[0,0]
        
geo_fence_exits = 3
coverage = 0.87
coverage_percent = coverage * 100

# Create a list of elements to add to the document
elements = []

# Add the title
title_style = getSampleStyleSheet()["Title"]
title = Paragraph("Safety Report Summary", title_style)
elements.append(title)

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

# Add a paragraph entry after the subtitle

paragraph = Paragraph("The Coverage report quanitifies the total coverage of the desired region to be mowed by the platform.", paragraph_style)
elements.append(paragraph)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

subtitle = Paragraph("Coverage Percent: (Dummy Amount) "+str(coverage_percent)+"%", subtitle_style)
elements.append(subtitle)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space

paragraph = Paragraph("The Mower Collisions Display the total accounts of collisions made with individual types of assets in the Environment.", paragraph_style)
elements.append(paragraph)

elements.append(Spacer(1, 24))  # 24 points is roughly equivalent to two lines of space



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

# Add an image to the top-left corner

#image = Image('/path/to/directory', width=72, height=72)  # Replace "your_image.png" with the path to your image
#elements.insert(0, image)
# Add an image to the top-left corner

# Build the PDF document
doc.build(elements)







