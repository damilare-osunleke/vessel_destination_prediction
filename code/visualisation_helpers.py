import folium
import h3
import pandas as pd


def visualise_trip(df_trip, all_ports):
    """
    Generates a folium map visualizing the ship's trip based on H3 hexagons, position coordinates, and port locations.

    Parameters:
    - df_trip (pd.DataFrame): Dataframe containing trip data with columns for position latitude ('pos_latitude'),
      position longitude ('pos_longitude'), last and next location codes ('locode_last', 'locode_next'), and H3 cell 
      identifiers ('cell_h3').
    - all_ports (pd.DataFrame): Dataframe containing information about ports with columns for port locode ('portlocode'), 
      latitude ('latitude'), and longitude ('longitude').

    Returns:
    - folium.Map: A map visualization including the trajectory of the trip, port markers, and H3 hexagonal cells.

    The function marks each position on the trip with serial numbers and hexagons, draws the path of the trip, and marks 
    the ports with their names. It ensures that each unique H3 hexagonal cell along the trip is visualized and linked with 
    the corresponding trajectory and port positions.
    """

    # get list of ports in trip
    ports_sample_1 = df_trip.drop_duplicates(subset=['locode_last'])["locode_last"].to_list()
    ports_sample_2 = df_trip.drop_duplicates(subset=['locode_next'])["locode_next"].to_list()
    ports_sample_1.extend(ports_sample_2)
    ports_sample = list(set(ports_sample_1))

    # get coordinates
    lat_sample = df_trip["pos_latitude"].to_list()
    long_sample = df_trip["pos_longitude"].to_list()
    pos_coord_sample = [(x,y) for x,y in zip(lat_sample, long_sample)] 
    cell_sample = df_trip["cell_h3"].to_list()

    # get the boundaries of the cells represented by the indexes
    index_boundaries_sample = [h3.h3_to_geo_boundary(index) for index in cell_sample]
    unique_index_boundaries_sample = pd.unique(index_boundaries_sample).tolist()

    # get coordinates for ports
    ports_sample_df = pd.DataFrame(ports_sample, columns=["ports"])
    ports_sample_df = pd.merge(left= ports_sample_df, right=all_ports[["portlocode","latitude","longitude"]], left_on= 'ports', right_on= 'portlocode', how= 'left')
    ports_lat_sample =ports_sample_df["latitude"].to_list()
    ports_lon_sample =ports_sample_df["longitude"].to_list()
    ports_coord_sample = [(x,y) for x,y in zip(ports_lat_sample, ports_lon_sample)]


    # Using Folium map, plot the hexagons of these indexes
    map_center = [0, 0]  # Center the map at (0, 0)
    folium_map = folium.Map(location=map_center, zoom_start=2)

    # Add hexagon polygons to the map
    for poly in unique_index_boundaries_sample:
        folium_poly = [[coord for coord in poly]]
        folium.Polygon(locations=folium_poly, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(folium_map)

    # Add coordinates to the map
    for i, coords in enumerate(pos_coord_sample, start= 1):
        # serial numbers
        folium.Marker(
            location=coords,
            icon=folium.DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 12pt; color : black">{i}</div>',
            )
        ).add_to(folium_map)

        # point markers
        folium.Marker(
            location=coords,
            # popup=f'Stop {s_n}',
            icon=folium.Icon(color='gray', icon='asterisk')
        ).add_to(folium_map)

        
    # Add ports to the map
    for ports, name  in zip(ports_coord_sample,ports_sample) :
        # add ports
        folium.Marker(
            location=ports,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(folium_map)

        # add port name
        folium.Marker(
            location=ports,
            icon=folium.DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 12pt; color : black">{name}</div>',
            )
        ).add_to(folium_map)  

    # Draw the polyline linking the coordinates (representing the vessel's path)
    folium.PolyLine(pos_coord_sample, color='black', weight=2.5, opacity=1).add_to(folium_map)


    return folium_map


