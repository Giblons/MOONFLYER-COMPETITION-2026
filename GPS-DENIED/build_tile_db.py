"""
Geospatial Tile Database Builder Utility.
Enables compilation of high-fidelity ground reference databases (tiles.db) from:
1. A raw orthomosaic ground image / satellite photo with known WGS84 corner boundaries
   (automatically slices, maps, and imports 256x256 tiles at a target zoom level).
2. Existing tile sets structured in standard Slippy Map directory schemas (zoom/x/y.png).

Allows offline VGPS testing using custom video feeds and actual flight locations.
"""

import os
import sys
import math
import sqlite3
import argparse
import logging
import cv2
import numpy as np

import config
from geospatial import wgs84_to_slippy, slippy_to_wgs84_boundary

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VGPS.DBBuilder")


def init_database(db_path):
    """
    Initializes a standard MBTiles SQLite database layout.
    """
    logger.info(f"Initializing database at {db_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
    
    # Create indexes to speed up tile lookups during high-speed flight
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS tile_index ON tiles (zoom_level, tile_column, tile_row)")
    
    conn.commit()
    conn.close()


def insert_tile(db_path, zoom, x, y, tile_img):
    """
    Encodes tile image to PNG and inserts it into the MBTiles table.
    Note: MBTiles uses TMS y-coordinates where tile_row = 2^zoom - 1 - y.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Compress tile to PNG byte stream
    _, img_encoded = cv2.imencode('.png', tile_img)
    tile_bytes = img_encoded.tobytes()
    
    # Convert Web Mercator y-axis index to TMS row index
    tile_row = (2 ** zoom) - 1 - y
    
    cursor.execute("""
        INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data)
        VALUES (?, ?, ?, ?)
    """, (zoom, x, tile_row, sqlite3.Binary(tile_bytes)))
    
    conn.commit()
    conn.close()


def build_db_from_tiles_folder(db_path, folder_path, zoom=config.MAP_ZOOM_LEVEL):
    """
    Imports structured tiles from a local folder.
    Expects structure:
      folder_path/x/y.png (or folder_path/zoom/x/y.png)
    """
    logger.info(f"Scanning directory {folder_path} for tiles...")
    init_database(db_path)
    
    imported_count = 0
    
    # Support both folder_path/x/y.png and folder_path/zoom/x/y.png
    base_dir = folder_path
    zoom_dir_exists = os.path.exists(os.path.join(folder_path, str(zoom)))
    if zoom_dir_exists:
        base_dir = os.path.join(folder_path, str(zoom))
        
    for x_str in os.listdir(base_dir):
        x_dir = os.path.join(base_dir, x_str)
        if not os.path.isdir(x_dir):
            continue
            
        try:
            x = int(x_str)
        except ValueError:
            continue
            
        for y_file in os.listdir(x_dir):
            if not (y_file.endswith('.png') or y_file.endswith('.jpg') or y_file.endswith('.jpeg')):
                continue
                
            try:
                y = int(os.path.splitext(y_file)[0])
            except ValueError:
                continue
                
            tile_path = os.path.join(x_dir, y_file)
            tile_img = cv2.imread(tile_path)
            
            if tile_img is None:
                logger.warning(f"Unable to read image: {tile_path}")
                continue
                
            # Ensure tile is strictly 256x256 as required by GIS databases
            if tile_img.shape[0] != 256 or tile_img.shape[1] != 256:
                tile_img = cv2.resize(tile_img, (256, 256), interpolation=cv2.INTER_AREA)
                
            insert_tile(db_path, zoom, x, y, tile_img)
            imported_count += 1
            
    logger.info(f"SUCCESS: Imported {imported_count} tiles into {db_path}.")


def build_db_from_orthophoto(db_path, image_path, lat_top, lon_left, lat_bottom, lon_right, zoom=config.MAP_ZOOM_LEVEL):
    """
    Slices a single large georeferenced orthomosaic ground photo into Mercator tiles
    and saves them in the SQLite tiles table.
    """
    logger.info(f"Opening orthomosaic image: {image_path}...")
    ortho_img = cv2.imread(image_path)
    
    if ortho_img is None:
        logger.error(f"Failed to read orthomosaic image at {image_path}.")
        sys.exit(1)
        
    ortho_h, ortho_w = ortho_img.shape[:2]
    logger.info(f"Orthophoto loaded successfully. Resolution: {ortho_w}x{ortho_h} pixels.")
    
    # Initialize the database
    init_database(db_path)
    
    # Calculate the overlapping tiles in the target area
    start_x, start_y = wgs84_to_slippy(lat_top, lon_left, zoom)
    end_x, end_y = wgs84_to_slippy(lat_bottom, lon_right, zoom)
    
    # Ensure indices are correct order
    x_min, x_max = min(start_x, end_x), max(start_x, end_x)
    y_min, y_max = min(start_y, end_y), max(start_y, end_y)
    
    logger.info(f"Targeting Slippy Map tile grid boundaries at Zoom={zoom}:")
    logger.info(f"  X-Index Range: {x_min} to {x_max}")
    logger.info(f"  Y-Index Range: {y_min} to {y_max}")
    
    total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    logger.info(f"Slicing orthophoto into {total_tiles} Mercator tiles...")
    
    sliced_count = 0
    
    # Map coordinate scales for spatial interpolation
    lon_range = lon_right - lon_left
    lat_range = lat_bottom - lat_top  # Usually negative
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            # Calculate WGS84 boundaries of this target tile
            tile_lat_top, tile_lon_left, tile_lat_bottom, tile_lon_right = slippy_to_wgs84_boundary(x, y, zoom)
            
            # Map WGS84 coordinates to normalized pixel coordinates on source image (0.0 to 1.0)
            px_left = (tile_lon_left - lon_left) / lon_range
            px_right = (tile_lon_right - lon_left) / lon_range
            py_top = (tile_lat_top - lat_top) / lat_range
            py_bottom = (tile_lat_bottom - lat_top) / lat_range
            
            # Convert normalized coordinates to actual source image pixel indices
            ix_left = int(px_left * ortho_w)
            ix_right = int(px_right * ortho_w)
            iy_top = int(py_top * ortho_h)
            iy_bottom = int(py_bottom * ortho_h)
            
            # Ensure correct coordinates sequence
            src_x1, src_x2 = max(0, min(ix_left, ix_right)), max(0, min(ix_left, ix_right))
            # Recalculate coordinates correctly
            src_x1 = int(min(px_left, px_right) * ortho_w)
            src_x2 = int(max(px_left, px_right) * ortho_w)
            src_y1 = int(min(py_top, py_bottom) * ortho_h)
            src_y2 = int(max(py_top, py_bottom) * ortho_h)
            
            # Crop bounds
            src_x1 = max(0, min(src_x1, ortho_w - 1))
            src_x2 = max(0, min(src_x2, ortho_w))
            src_y1 = max(0, min(src_y1, ortho_h - 1))
            src_y2 = max(0, min(src_y2, ortho_h))
            
            # Skip if slice falls completely outside the reference image bounds
            if (src_x2 - src_x1) < 2 or (src_y2 - src_y1) < 2:
                continue
                
            # Crop tile from source image
            tile_slice = ortho_img[src_y1:src_y2, src_x1:src_x2]
            
            # Resize tile to strict 256x256
            tile_256 = cv2.resize(tile_slice, (256, 256), interpolation=cv2.INTER_AREA)
            
            # Insert tile to database
            insert_tile(db_path, zoom, x, y, tile_256)
            sliced_count += 1
            
    logger.info(f"SUCCESS: Successfully generated {sliced_count}/{total_tiles} tiles and saved in {db_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile a standard MBTiles SQLite reference database for Vision-Based GPS navigation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--db", 
        default=config.DATABASE_PATH,
        help="Target output database filepath (default: config.DATABASE_PATH)"
    )
    
    parser.add_argument(
        "--mode", 
        required=True, 
        choices=["ortho", "folder"],
        help="Creation mode:\n"
             " 'ortho'  - Slice a single high-res georeferenced orthophoto image.\n"
             " 'folder' - Import an existing directory structure of slippy map tiles."
    )
    
    # Parameters for 'ortho' mode
    parser.add_argument("--image", help="Path to high-resolution orthophoto image (required for 'ortho')")
    parser.add_argument("--lat-top", type=float, help="WGS84 latitude of Top-Left corner of orthomosaic (required for 'ortho')")
    parser.add_argument("--lon-left", type=float, help="WGS84 longitude of Top-Left corner of orthomosaic (required for 'ortho')")
    parser.add_argument("--lat-bottom", type=float, help="WGS84 latitude of Bottom-Right corner of orthomosaic (required for 'ortho')")
    parser.add_argument("--lon-right", type=float, help="WGS84 longitude of Bottom-Right corner of orthomosaic (required for 'ortho')")
    
    # Parameters for 'folder' mode
    parser.add_argument("--folder", help="Path to the directory containing pre-downloaded tiles (required for 'folder')")
    
    parser.add_argument(
        "--zoom", 
        type=int, 
        default=config.MAP_ZOOM_LEVEL,
        help=f"Target Mercator zoom level (default: {config.MAP_ZOOM_LEVEL})"
    )

    args = parser.parse_args()
    
    if args.mode == "ortho":
        if not all([args.image, args.lat_top is not None, args.lon_left is not None, 
                    args.lat_bottom is not None, args.lon_right is not None]):
            parser.error("Mode 'ortho' requires: --image, --lat-top, --lon-left, --lat-bottom, --lon-right.")
        
        build_db_from_orthophoto(
            db_path=args.db,
            image_path=args.image,
            lat_top=args.lat_top,
            lon_left=args.lon_left,
            lat_bottom=args.lat_bottom,
            lon_right=args.lon_right,
            zoom=args.zoom
        )
        
    elif args.mode == "folder":
        if not args.folder:
            parser.error("Mode 'folder' requires: --folder.")
            
        build_db_from_tiles_folder(
            db_path=args.db,
            folder_path=args.folder,
            zoom=args.zoom
        )
