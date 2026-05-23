"""
Geospatial Calculations and MBTiles Integration Module.
Implements WGS84 ellipsoidal transformations to Web Mercator Slippy Map tiles,
robust SQL query access to locally cached MBTiles SQLite schemas,
and affine standard World File projection equations converting local pixels to global WGS84 coordinates.
"""

import os
import math
import sqlite3
import logging
import cv2
import numpy as np
import config

logger = logging.getLogger("VGPS.Geospatial")


def wgs84_to_slippy(lat, lon, zoom):
    """
    Converts WGS84 coordinates to standard Web Mercator Slippy Map tile indices (x, y).
    Uses standard flooring logic and trigonometric projections.
    """
    # Clip coordinates to valid Web Mercator limits
    lat = max(min(lat, 85.0511), -85.0511)
    lon = max(min(lon, 180.0), -180.0)
    
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    
    return x_tile, y_tile


def slippy_to_wgs84_boundary(x, y, zoom):
    """
    Calculates the exact WGS84 boundaries of a Slippy Map tile.
    Returns:
        lat_top (float), lon_left (float), lat_bottom (float), lon_right (float)
    """
    n = 2.0 ** zoom
    
    # Upper-left coordinates
    lon_left = x / n * 360.0 - 180.0
    lat_rad_top = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat_top = math.degrees(lat_rad_top)
    
    # Lower-right coordinates
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_rad_bottom = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))
    lat_bottom = math.degrees(lat_rad_bottom)
    
    return lat_top, lon_left, lat_bottom, lon_right


def initialize_mock_mbtiles_db(db_path=config.DATABASE_PATH):
    """
    Automatically creates and initializes an SQLite/MBTiles database if missing.
    Populates it with high-contrast, structural reference tiles matching the takeoff point,
    allowing the system to run out-of-the-box and verify features.
    """
    if os.path.exists(db_path):
        logger.info(f"Geospatial MBTiles database found at {db_path}.")
        return

    logger.warning(f"MBTiles database not found at {db_path}. Initializing self-contained mock cache database...")
    
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create standard MBTiles tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tiles (
                zoom_level INTEGER,
                tile_column INTEGER,
                tile_row INTEGER,
                tile_data BLOB
            )
        """)
        
        # Populate standard metadata
        metadata = [
            ("name", "VGPS Mock Database"),
            ("type", "baselayer"),
            ("version", "1.0.0"),
            ("description", "Mock ground tiles for VGPS verification"),
            ("format", "png"),
            ("bounds", f"-180.0,-85.0,180.0,85.0")
        ]
        cursor.executemany("INSERT INTO metadata VALUES (?, ?)", metadata)
        
        # Generate and insert mock tile corresponding to config.TAKEOFF_LAT / LON
        x_tile, y_tile = wgs84_to_slippy(config.TAKEOFF_LAT, config.TAKEOFF_LON, config.MAP_ZOOM_LEVEL)
        
        # Generate reference tile matching the synthetic drone capture view
        # The tile size is 256x256 pixels
        tile_size = 256
        ref_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        ref_tile[:, :] = [34, 139, 34]  # Forest Green
        
        # Draw static fields (landmarks)
        # These match the coordinates of landmarks in the mock camera generator
        # but at a global reference tile perspective
        np.random.seed(42)
        field_centers = [
            (40, 40, [60, 179, 113]),    # Medium Sea Green
            (180, 60, [107, 142, 35]),   # Olive Drab
            (60, 192, [143, 188, 143]),  # Dark Sea Green
            (200, 180, [85, 107, 47]),   # Dark Olive Green
            (128, 128, [46, 139, 87])    # Sea Green
        ]
        
        for cx, cy, color in field_centers:
            cv2.rectangle(ref_tile, (cx - 32, cy - 32), (cx + 32, cy + 32), color, -1)
            cv2.rectangle(ref_tile, (cx - 32, cy - 32), (cx + 32, cy + 32), (200, 200, 200), 2)
            cv2.circle(ref_tile, (cx, cy), 4, (100, 100, 255), -1)
            
        # Draw central road landmark matching camera
        cv2.line(ref_tile, (0, 120), (tile_size, 120), (128, 128, 128), 10)
        cv2.line(ref_tile, (0, 120), (tile_size, 120), (255, 255, 255), 1)
        
        # Compress to PNG byte stream
        _, img_encoded = cv2.imencode('.png', ref_tile)
        tile_bytes = img_encoded.tobytes()
        
        # In MBTiles schema (TMS standard), y-axis is inverted:
        # tile_row = 2^zoom - 1 - y_tile
        tile_row = (2 ** config.MAP_ZOOM_LEVEL) - 1 - y_tile
        
        cursor.execute(
            "INSERT INTO tiles VALUES (?, ?, ?, ?)",
            (config.MAP_ZOOM_LEVEL, x_tile, tile_row, sqlite3.Binary(tile_bytes))
        )
        
        conn.commit()
        conn.close()
        logger.info("Mock MBTiles SQLite database initialized and populated with takeoff tile.")
    except Exception as e:
        logger.error(f"Failed to initialize mock database: {e}")


class GeospatialDatabase:
    """
    Encapsulates interaction with local MBTiles cache.
    Resolves geographic queries and loads high-resolution reference tiles.
    """
    def __init__(self, db_path=config.DATABASE_PATH):
        self.db_path = db_path
        initialize_mock_mbtiles_db(self.db_path)

    def fetch_tile_by_wgs84(self, lat, lon, zoom=config.MAP_ZOOM_LEVEL):
        """
        Calculates active slippy indices, queries the SQL schema, and returns the reference image.
        Returns:
            tile_image (np.ndarray 256x256 BGR), tile_x (int), tile_y (int)
        """
        x_tile, y_tile = wgs84_to_slippy(lat, lon, zoom)
        
        # Convert to TMS Row index
        tile_row = (2 ** zoom) - 1 - y_tile
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?",
                (zoom, x_tile, tile_row)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row is not None:
                tile_blob = row[0]
                tile_array = np.frombuffer(tile_blob, np.uint8)
                # Decode byte array to BGR OpenCV image
                tile_image = cv2.imdecode(tile_array, cv2.IMREAD_COLOR)
                if tile_image is not None:
                    return tile_image, x_tile, y_tile
            
            logger.warning(f"Tile x={x_tile}, y={y_tile} not found in database. Retrying database initialization...")
            return None, x_tile, y_tile
            
        except Exception as e:
            logger.error(f"Error querying tile database: {e}")
            return None, x_tile, y_tile


class CoordinateTransformer:
    """
    Performs standard World File Affine coordinate transformation to project 
    localized pixel matching estimates into absolute global WGS84 Coordinates.
    """
    def __init__(self, tile_x, tile_y, zoom=config.MAP_ZOOM_LEVEL, tile_size=256.0):
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.zoom = zoom
        self.tile_size = tile_size
        
        # Calculate boundaries of the Slippy tile in global degrees
        self.lat_top, self.lon_left, self.lat_bottom, self.lon_right = slippy_to_wgs84_boundary(tile_x, tile_y, zoom)
        
        # Compute World File coefficients:
        # A: longitude scale per pixel (positive)
        # E: latitude scale per pixel (negative, as pixel y-index grows downwards, latitude decreases)
        self.A = (self.lon_right - self.lon_left) / self.tile_size
        self.E = (self.lat_bottom - self.lat_top) / self.tile_size
        
        # Rotation and Skew are zero
        self.B = 0.0
        self.D = 0.0
        
        # Translation offsets (top-left pixel coordinates correspond to lon_left, lat_top)
        self.C = self.lon_left
        self.F = self.lat_top

    def pixel_to_wgs84(self, px, py):
        """
        Applies standard World File affine conversion equations:
        mx = C + px * A + py * B
        my = F + px * D + py * E
        where B=0, D=0 (no skew or rotation).
        
        Returns:
            lat (float), lon (float)
        """
        lon = self.C + (px * self.A)
        lat = self.F + (py * self.E)
        
        # Constrain boundary limits
        lon = max(min(lon, 180.0), -180.0)
        lat = max(min(lat, 90.0), -90.0)
        
        return lat, lon
