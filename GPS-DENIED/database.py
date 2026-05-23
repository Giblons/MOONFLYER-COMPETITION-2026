import os
import json
import sqlite3
import numpy as np
import cv2
import math

class DatabaseManager:
    """
    Multi-Tiered Database Integration & Expanding Radius Search Protocol.
    Queries local databases (MBTiles for raster, GeoJSON for vector).
    Implements an Expanding Radius Search based on "last known good" coordinate.
    """
    def __init__(self, mbtiles_path=None, geojson_path=None):
        self.mbtiles_path = mbtiles_path
        self.geojson_path = geojson_path
        self.db_conn = None
        self.vector_data = None

        self.connect_mbtiles()
        self.load_geojson()

    def connect_mbtiles(self):
        """Connects to the MBTiles (SQLite) database."""
        if self.mbtiles_path and os.path.exists(self.mbtiles_path):
            try:
                self.db_conn = sqlite3.connect(self.mbtiles_path, check_same_thread=False)
                print(f"[DatabaseManager] Connected to MBTiles at {self.mbtiles_path}")
            except sqlite3.Error as e:
                print(f"[DatabaseManager] MBTiles connection error: {e}")

    def load_geojson(self):
        """Loads vector data from a GeoJSON file."""
        if self.geojson_path and os.path.exists(self.geojson_path):
            try:
                with open(self.geojson_path, 'r') as f:
                    self.vector_data = json.load(f)
                print(f"[DatabaseManager] Loaded GeoJSON from {self.geojson_path}")
            except Exception as e:
                print(f"[DatabaseManager] GeoJSON loading error: {e}")

    def get_tile(self, zoom, tile_column, tile_row):
        """
        Retrieves a specific tile from MBTiles database.
        Returns the tile as a decoded OpenCV image.
        """
        if not self.db_conn:
            return None

        # In MBTiles, the tile_row is inverted (TMS format vs XYZ)
        tms_row = (1 << zoom) - 1 - tile_row

        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (zoom, tile_column, tms_row)
        )
        row = cursor.fetchone()

        if row and row[0]:
            tile_data = row[0]
            # Decode the image bytes to OpenCV format
            nparr = np.frombuffer(tile_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        return None

    def lonlat_to_tile(self, lon, lat, zoom):
        """Converts WGS84 coordinates to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    def tile_to_lonlat(self, xtile, ytile, zoom):
        """Converts tile coordinates to WGS84 coordinates (top-left corner)."""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lon_deg, lat_deg

    def expanding_radius_search(self, last_lon, last_lat, zoom, radius=1, max_radius=10):
        """
        Expanding Radius Search Protocol.
        Yields tiles around the last known location, expanding outwards.
        Yields tuple: (xtile, ytile, tile_image)
        """
        base_x, base_y = self.lonlat_to_tile(last_lon, last_lat, zoom)

        current_radius = 0
        searched = set()

        while current_radius <= max_radius:
            # Generate coordinates for current radius ring
            for dx in range(-current_radius, current_radius + 1):
                for dy in range(-current_radius, current_radius + 1):
                    # Only yield tiles exactly at the current radius boundary
                    if max(abs(dx), abs(dy)) == current_radius:
                        x = base_x + dx
                        y = base_y + dy

                        if (x, y) not in searched:
                            searched.add((x, y))
                            tile_img = self.get_tile(zoom, x, y)
                            if tile_img is not None:
                                yield (x, y, tile_img)
            current_radius += 1

    def query_vector_data(self, min_lon, min_lat, max_lon, max_lat):
        """
        Queries the loaded GeoJSON vector data for geometries intersecting a bounding box.
        """
        results = []
        if not self.vector_data or 'features' not in self.vector_data:
            return results

        for feature in self.vector_data['features']:
            # Simplified bounding box check for vectors
            # In a real system, this would use an R-Tree or spatial index
            geom = feature.get('geometry')
            if geom and geom.get('type') in ['Polygon', 'MultiPolygon', 'LineString']:
                coords = geom.get('coordinates')
                # A very basic check: if any coordinate is within bounds, include it
                # For a production system, use Shapely for exact intersection
                try:
                    if self._check_intersection(coords, min_lon, min_lat, max_lon, max_lat):
                        results.append(feature)
                except Exception as e:
                    pass
        return results

    def _check_intersection(self, coords, min_lon, min_lat, max_lon, max_lat):
        # Recursive check for nested coordinates (Polygons, MultiPolygons)
        if isinstance(coords[0], (int, float)):
            lon, lat = coords[0], coords[1]
            return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
        else:
            for sub_coord in coords:
                if self._check_intersection(sub_coord, min_lon, min_lat, max_lon, max_lat):
                    return True
        return False

    def close(self):
        """Closes the database connection."""
        if self.db_conn:
            self.db_conn.close()
