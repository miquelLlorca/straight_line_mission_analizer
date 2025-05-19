import gpxpy
import pandas as pd
import folium
import os
import webbrowser
from pyproj import Geod
from math import radians, degrees, atan2, sin, cos
import streamlit as st
from streamlit.components.v1 import html


GEOD = Geod(ellps='WGS84')


def get_color_from_dist(dist):
    if(dist<10):
        return 'blue'
    if(dist<25):
        return 'green'
    if(dist<50):
        return 'yellow'
    if(dist<100):
        return 'orange'
    if(dist<250):
        return 'red'
    return 'black'



class StraighLineAnalisys:
    m: folium.Map
    df: pd.DataFrame

    def __init__(self, path, line):
        self.real_gpx_path = path
        self.map_path = os.path.join(os.getcwd(),'maps',f'{os.path.basename(self.real_gpx_path)}.html')
        self.start = [self.__dms_to_dd(dms) for dms in line[0]]
        self.end = [self.__dms_to_dd(dms) for dms in line[1]]

    def __dms_to_dd(self, dms_str):
        dms_str = dms_str.strip()
        deg, min_, sec_dir = dms_str[:-1].split('°'), '', ''
        if "'" in deg[1]:
            min_, sec_dir = deg[1].split("'")
        sec = sec_dir.replace('"', '').strip('NSEO ')
        dir_ = dms_str[-1]

        dd = float(deg[0]) + float(min_) / 60 + float(sec) / 3600
        if dir_ in ['S', 'O', 'W']:
            dd *= -1
        return dd

    def __geodesic_cross_track_distance(self, point):
        """
        All points are (lat, lon). Returns distance in meters.
        """
        lat_p, lon_p = point
        lat1, lon1 = self.start
        lat2, lon2 = self.end
        
        # Azimuth and distance along the line
        az12, az21, dist12 = GEOD.inv(lon1, lat1, lon2, lat2)
        az13, az31, dist13 = GEOD.inv(lon1, lat1, lon_p, lat_p)

        # Angular distance and azimuth difference
        delta13 = dist13 / GEOD.a
        theta = radians(az13 - az12)

        # Cross-track distance
        xtd = sin(delta13) * sin(theta) * GEOD.a

        # Along-track distance
        atd = atan2(
            sin(delta13) * cos(theta),
            cos(delta13)
        ) * GEOD.a

        # Destination point at atd meters along the line
        proj_lon, proj_lat, _ = GEOD.fwd(lon1, lat1, az12, atd)

        return abs(xtd), proj_lat, proj_lon


    def __load_gpx_as_df(self):
        with open(self.real_gpx_path, 'r') as f:
            gpx = gpxpy.parse(f)

        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for p in segment.points:
                    points.append({
                        'time': p.time,
                        'lat': p.latitude,
                        'lon': p.longitude,
                        'elevation': p.elevation
                    })

        self.df = pd.DataFrame(points)

    def __init_map(self):
        # Center of map (first point in your df)
        self.m = folium.Map(location=[self.df.lat[0], self.df.lon[0]], zoom_start=14)
        
    def __plot_gpx_on_map(self):
        # Add the GPS track
        folium.PolyLine(self.df[['lat', 'lon']].values, color='green', weight=3).add_to(self.m)

        # Optional: add markers
        folium.Marker([self.df.lat[0], self.df.lon[0]], tooltip="Start").add_to(self.m)
        folium.Marker([self.df.lat.iloc[-1], self.df.lon.iloc[-1]], tooltip="End").add_to(self.m)

    def __plot_line_on_map(self):
        folium.PolyLine([self.start, self.end], color='blue', weight=2.5).add_to(self.m)

    def __get_distances(self):
        dists = []
        proj_points = []
        step = 20
        for i, row in self.df.iterrows():
            if(i%step==0):
                dist, proj_lat, proj_lon = self.__geodesic_cross_track_distance([row['lat'], row['lon']])
                dists.append(dist)
                proj_points.append((proj_lat, proj_lon))
                # draw a line from the point to the line perpendicularly
        return pd.DataFrame({
            'lat': self.df.iloc[::step]['lat'].values,
            'lon': self.df.iloc[::step]['lon'].values,
            'deviation': dists,
            'proj_lat': [p[0] for p in proj_points],
            'proj_lon': [p[1] for p in proj_points]
        })


    def __plot_dist_lines(self, df:pd.DataFrame):
        for _, row in df.iterrows():
            folium.PolyLine(
                [(row['lat'], row['lon']), (row['proj_lat'], row['proj_lon'])],
                color=get_color_from_dist(row['deviation']),
                weight=2,
                opacity=1
            ).add_to(self.m)

    def __get_metrics(self, dist_df):
        devs = dist_df['deviation']
        print("Median deviation:", devs.median())
        print("Mean deviation:", devs.mean())
        print("Max deviation:", devs.max())
        print("Std deviation:", devs.std())
        print("95th percentile:", devs.quantile(0.95))
    
    def proccess_gpx(self):
        self.__load_gpx_as_df()
        self.__init_map()
        self.__plot_gpx_on_map()
        self.__plot_line_on_map()
        dist_df = self.__get_distances()
        self.__plot_dist_lines(dist_df)
        self.__get_metrics(dist_df)


    def save_map(self):
        self.m.save(self.map_path)
        # webbrowser.open(self.map_path)
        html(self.m._repr_html_(), height=900)

challenges = [
    {
        'real_gpx_path': 'gpxs\Perdenme_per_la_Ermita.gpx',
        'line': [['38°30\'60.00"N','0°15\'28.91"O'], 
                 ['38°30\'10.41"N','0°21\'23.66"O']]
    },{
        'real_gpx_path': 'gpxs\Passeig_carrascar.gpx',
        'line': [['38°37\'39.8"N','0°12\'42.5"W'], 
                 ['38°37\'13.2"N','0°14\'03.6"W']]
    }
]

st.set_page_config(layout="wide")
st.title("Straight Line Challenge Analyzer")

challenge = 1
real_gpx_path = challenges[challenge]['real_gpx_path']
line = challenges[challenge]['line']


analiser = StraighLineAnalisys(real_gpx_path, line)
analiser.proccess_gpx()
analiser.save_map()

