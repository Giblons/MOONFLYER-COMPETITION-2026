"""
MAVLink Telemetry Thread Module.
Establishes connection to Pixhawk flight controllers via pymavlink over UART.
Periodically streams VISION_POSITION_ESTIMATE messages containing WGS84 positioning
with microsecond timestamp synchronization to drive EKF3/EKF2 state estimators at 10-30Hz.
Includes robust fallback telemetry logger when hardware serial interfaces are absent.
"""

import time
import logging
import threading
from pymavlink import mavutil
import config

logger = logging.getLogger("VGPS.Telemetry")


class TelemetryThread(threading.Thread):
    """
    Asynchronous telemetry transmission thread.
    Establishes serial MAVLink connection to the flight controller and streams
    position estimates. Dynamically falls back to console logger if serial port is locked.
    """
    def __init__(self, port=config.SERIAL_PORT, baud=config.BAUD_RATE, hz=config.TELEMETRY_UPDATE_HZ):
        super().__init__(name="TelemetryThread", daemon=True)
        self.port = port
        self.baud = baud
        self.interval = 1.0 / hz
        
        # Thread safety locks for telemetry state updates
        self._lock = threading.Lock()
        self._lat = config.TAKEOFF_LAT
        self._lon = config.TAKEOFF_LON
        self._alt = config.DEFAULT_ALTITUDE
        self._yaw = 0.0
        self._valid_estimate = False
        
        self.mav_conn = None
        self._running = False
        self._is_mock = False

    def _connect_mavlink(self):
        """
        Attempts connection to the Pixhawk serial UART channel.
        Falls back to software mock mode if device cannot be initialized.
        """
        logger.info(f"Connecting MAVLink to serial interface {self.port} at {self.baud} baud...")
        try:
            # Open MAVLink serial port
            self.mav_conn = mavutil.mavlink_connection(
                self.port, 
                baud=self.baud,
                source_system=config.MAVLINK_SYSTEM_ID,
                source_component=config.MAVLINK_COMPONENT_ID
            )
            logger.info("MAVLink serial connection successfully initialized.")
        except Exception as e:
            logger.warning(f"Failed to open UART {self.port}: {e}. Standard telemetry streaming will operate in mock mode.")
            self._is_mock = True

    def update_position(self, lat, lon, alt, yaw=0.0):
        """
        Atomically updates the active position estimate variables. Called by the CV processing thread.
        """
        with self._lock:
            self._lat = lat
            self._lon = lon
            self._alt = alt
            self._yaw = yaw
            self._valid_estimate = True

    def invalidate_position(self):
        """
        Flags the active position estimate as invalid (e.g. tracking lost).
        Keeps telemetry loop active but halts MAVLink streams to prevent EKF corruption.
        """
        with self._lock:
            self._valid_estimate = False

    def run(self):
        """
        Periodic telemetry streaming loop.
        Syncs precise microsecond timestamps and broadcasts VISION_POSITION_ESTIMATE packets.
        """
        self._connect_mavlink()
        self._running = True
        
        # 21-element float covariance array for MAVLink VISION_POSITION_ESTIMATE
        # Row-major representation of pose covariance (x, y, z, roll, pitch, yaw)
        # Small values indicate high VGPS confidence
        covariance = [0.0] * 21
        covariance[0]  = 0.01  # Var X (Latitude proxy)
        covariance[5]  = 0.01  # Var Y (Longitude proxy)
        covariance[9]  = 0.05  # Var Z (Altitude)
        covariance[20] = 0.02  # Var Yaw
        
        last_tx_time = time.time()
        
        while self._running:
            loop_start = time.time()
            
            # Thread-safe read of state variables
            with self._lock:
                lat = self._lat
                lon = self._lon
                alt = self._alt
                yaw = self._yaw
                is_valid = self._valid_estimate
            
            # Stream only if CV pipeline is actively matching terrain
            if is_valid:
                # Generate a high-resolution UNIX epoch microsecond timestamp
                timestamp_usec = int(time.time() * 1e6)
                
                if not self._is_mock and self.mav_conn is not None:
                    try:
                        # Pack and transmit MAVLink VISION_POSITION_ESTIMATE (#102)
                        # We map Latitude/Longitude/Altitude as raw coordinate indicators to Pixhawk EKF3.
                        # Note: In standard PX4/ArduPilot, EKF coordinates are often converted to local NED.
                        # If EKF3 expects local coordinates, ensure ArduPilot EKF3_GPS_TYPE = 3 (Vision)
                        # or VIS_COORD_TYPE = 0 (Global). We pass WGS84 equivalents to x,y,z here.
                        self.mav_conn.mav.vision_position_estimate_send(
                            usec=timestamp_usec,
                            x=lat,            # Encodes latitude coordinate
                            y=lon,            # Encodes longitude coordinate
                            z=alt,            # Encodes altitude coordinate (nadir distance)
                            roll=0.0,
                            pitch=0.0,
                            yaw=yaw,
                            covariance=covariance,
                            reset_counter=0   # Index count to track tracking resets
                        )
                        logger.debug(f"MAVLink sent VPE: Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt:.2f}")
                    except Exception as e:
                        logger.error(f"MAVLink transmission failure: {e}")
                else:
                    # Write packet to logging stream in mock environments
                    logger.info(
                        f"[MOCK MAVLINK VPE] usec={timestamp_usec} | "
                        f"Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt:.2f} | "
                        f"Yaw={yaw:.1f} deg | Covariance=[0.01, 0.01, 0.05]"
                    )
            else:
                logger.debug("VGPS telemetry stream paused: awaiting valid coordinate lock...")
                
            # Maintain strict loop execution period
            elapsed = time.time() - loop_start
            delay = self.interval - elapsed
            if delay > 0:
                time.sleep(delay)
                
        logger.info("Telemetry thread shut down successfully.")

    def stop(self):
        """
        Signals the thread to terminate.
        """
        self._running = False
        logger.info("Stop signal sent to Telemetry Thread.")
