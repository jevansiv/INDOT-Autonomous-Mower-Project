import pyproj

def convert_coordinates(latitude, longitude):
    # Define the projection from geographic to UTM Zone 16N (NAD83)
    geographic = pyproj.CRS("EPSG:4326")
    utm_zone_16n = pyproj.CRS("EPSG:26916")
    
    # Create a transformer
    transformer = pyproj.Transformer.from_crs(geographic, utm_zone_16n, always_xy=True)
    
    # Transform the coordinates
    projected_x, projected_y = transformer.transform(longitude, latitude)
    
    return projected_x, projected_y

# Given coordinates
latitude = 40.4 # update coords
longitude = -86.9 # update coords

# Convert to projected coordinates (UTM Zone 16N)
projected_x, projected_y = convert_coordinates(latitude, longitude)

print(f"Latitude: {latitude}, Longitude: {longitude}")
print(f"Projected X: {projected_x}, Projected Y: {projected_y}")