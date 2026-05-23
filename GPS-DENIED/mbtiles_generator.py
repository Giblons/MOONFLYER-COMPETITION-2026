import sqlite3
import requests
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
# You will need to install these: pip install mercantile tqdm
import mercantile
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Center of the circle (From your Google Maps pin)
# NOTE: Please double check these coordinates! 
# (Right-click the pin on Google Maps and copy the numbers if these aren't exact)
LATITUDE = 23.725018654310063
LONGITUDE = 120.3742054423277
RADIUS_KM = 5.0     # 5km for training
MIN_ZOOM = 1
MAX_ZOOM = 20

DB_NAME = "uav_vision_map(1).mbtiles"

# Google Maps Satellite (Often more up-to-date, used on Soar Atlas)
TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

# Headers to simulate a browser and avoid getting blocked by the map provider
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_bounding_box(lat, lon, radius_km):
    """Calculates the North, South, East, West bounding box for a given radius."""
    # 1 degree of latitude is ~111.32 km
    lat_delta = radius_km / 111.32
    # 1 degree of longitude is ~111.32 km * cos(latitude)
    lon_delta = radius_km / (111.32 * math.cos(math.radians(lat)))
    
    south = lat - lat_delta
    north = lat + lat_delta
    west = lon - lon_delta
    east = lon + lon_delta
    return west, south, east, north

def setup_mbtiles_db(db_name):
    """Initializes the MBTiles SQLite database schema."""
    if os.path.exists(db_name):
        print(f"Database {db_name} already exists. Resuming/Appending...")
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # MBTiles specification tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            name TEXT,
            value TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tiles (
            zoom_level INTEGER,
            tile_column INTEGER,
            tile_row INTEGER,
            tile_data BLOB,
            UNIQUE (zoom_level, tile_column, tile_row)
        );
    """)
    
    # Insert required metadata
    cursor.execute("INSERT OR IGNORE INTO metadata (name, value) VALUES ('name', 'UAV_Vision_DB')")
    cursor.execute("INSERT OR IGNORE INTO metadata (name, value) VALUES ('type', 'baselayer')")
    cursor.execute("INSERT OR IGNORE INTO metadata (name, value) VALUES ('version', '1.1')")
    cursor.execute("INSERT OR IGNORE INTO metadata (name, value) VALUES ('description', 'Offline satellite imagery for UAV CV matching')")
    cursor.execute("INSERT OR IGNORE INTO metadata (name, value) VALUES ('format', 'jpg')")
    
    conn.commit()
    return conn

def download_tile(tile, total_retries=3):
    """Downloads a single tile from the provider."""
    z, x, y = tile.z, tile.x, tile.y
    url = TILE_URL.format(z=z, y=y, x=x)
    
    # MBTiles uses TMS indexing (Y goes up from the bottom). Slippy maps Y goes down.
    # We must flip the Y coordinate for the database!
    tms_y = (2**z - 1) - y
    
    for attempt in range(total_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return (z, x, tms_y, response.content)
            elif response.status_code == 404:
                return None # Blank tile / Ocean / No data
            else:
                time.sleep(1) # Back off if rate limited
        except Exception:
            time.sleep(1)
            
    return None

def main():
    print(f"Calculating map tiles for {RADIUS_KM}km radius around {LATITUDE}, {LONGITUDE}...")
    bbox = get_bounding_box(LATITUDE, LONGITUDE, RADIUS_KM)
    
    tiles_to_download = []
    for zoom in range(MIN_ZOOM, MAX_ZOOM + 1):
        # Generate all tile coordinates within our bounding box for this zoom level
        tiles = list(mercantile.tiles(*bbox, zooms=[zoom]))
        tiles_to_download.extend(tiles)
        print(f"Zoom Level {zoom}: {len(tiles)} tiles required.")

    total_tiles = len(tiles_to_download)
    print(f"\nTotal tiles to process: {total_tiles}")
    
    conn = setup_mbtiles_db(DB_NAME)
    cursor = conn.cursor()
    
    # Check which tiles we already have so we can resume if the script is stopped
    cursor.execute("SELECT zoom_level, tile_column, tile_row FROM tiles")
    existing_tiles = set(cursor.fetchall())
    
    # Filter out already downloaded tiles (remembering to flip Y for the check)
    pending_tiles = [
        t for t in tiles_to_download 
        if (t.z, t.x, (2**t.z - 1) - t.y) not in existing_tiles
    ]
    
    print(f"Skipping {total_tiles - len(pending_tiles)} already downloaded tiles.")
    print(f"Beginning download of {len(pending_tiles)} tiles...\n")

    # Use multi-threading to speed up downloads (5 workers to avoid getting IP banned)
    successful = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_tile, tile): tile for tile in pending_tiles}
        
        for future in tqdm(as_completed(futures), total=len(pending_tiles), desc="Downloading Tiles"):
            result = future.result()
            if result:
                z, x, tms_y, image_data = result
                cursor.execute(
                    "INSERT OR IGNORE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
                    (z, x, tms_y, image_data)
                )
                successful += 1
                
                # Commit to database every 100 successful downloads
                if successful % 100 == 0:
                    conn.commit()

    conn.commit()
    conn.close()
    print(f"\nFinished! Database saved as {DB_NAME}")
    print("This file can now be copied to your RADXA Zero 3W for offline flight!")

if __name__ == "__main__":
    main()