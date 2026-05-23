import socket
import serial
import datetime
import struct
from fastcrc import crc8
from pymavlink import mavutil

FRAME_TYPE = "QUADCOPTER"
UDP_IP = "0.0.0.0"       # Listen on all interfaces (crucial for Radxa to receive remotely)
UDP_PORT = 13131
GPS_PORT = "COM5"
GPS_BAUD = 115200
MAVLINK_PORT = "COM6"
MAVLINK_BAUD = 115200

def decimal_to_nmea(lat, lon):
    # Latitude
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60
    lat_nmea = f"{lat_deg:02d}{lat_min:07.4f}"
    lat_dir = "N" if lat >= 0 else "S"

    # Longitude
    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60
    lon_nmea = f"{lon_deg:03d}{lon_min:07.4f}"
    lon_dir = "E" if lon >= 0 else "W"

    return lat_nmea, lat_dir, lon_nmea, lon_dir

def checksum(nmea_sentence):
    cksum = 0
    for char in nmea_sentence:
        cksum ^= ord(char)
    return f"{cksum:02X}"

def create_gga(lat, lon, alt = 0.0):
    now = datetime.datetime.now(datetime.timezone.utc)
    time_str = now.strftime("%H%M%S")

    lat_nmea, lat_dir, lon_nmea, lon_dir = decimal_to_nmea(lat, lon)

    sentence = f"GPGGA,{time_str},{lat_nmea},{lat_dir},{lon_nmea},{lon_dir},1,10,1.0,{alt},M,0.0,M,,"
    return f"${sentence}*{checksum(sentence)}"

def create_rmc(lat, lon, speed = 0.0):
    now = datetime.datetime.now(datetime.timezone.utc)
    time_str = now.strftime("%H%M%S")
    date_str = now.strftime("%d%m%y")

    lat_nmea, lat_dir, lon_nmea, lon_dir = decimal_to_nmea(lat, lon)

    sentence = f"GPRMC,{time_str},A,{lat_nmea},{lat_dir},{lon_nmea},{lon_dir},{speed:.1f},0.0,{date_str},,,A"
    return f"${sentence}*{checksum(sentence)}"

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def send_rc(roll, pitch, throttle, yaw, m: mavutil.mavserial = None):
    if m is None:
        return
    m.mav.rc_channels_override_send(
        m.target_system,     # target system
        m.target_component,   # target component
        roll,     # CH1
        pitch,    # CH2
        throttle, # CH3
        yaw,      # CH4
        0, 0, 0, 0  # CH5–CH8 (0 = ignore)
    )
    return

def process_gps_data(data: bytes, ser: serial.Serial = None):
    unpacked_payload = struct.unpack('!Bff', data)
    print(f"Received Legacy GPS: Lat: {unpacked_payload[1]} Lon: {unpacked_payload[2]}")

    gga = create_gga(unpacked_payload[1], unpacked_payload[2])
    rmc = create_rmc(unpacked_payload[1], unpacked_payload[2])

    if ser is not None:
        ser.write((gga + "\r\n").encode())
        ser.write((rmc + "\r\n").encode())

    print(gga)
    print(rmc)
    return

def process_vgps_telemetry(data: bytes, ser: serial.Serial = None):
    """
    Parses high-precision VGPS telemetry packets (type 0x10).
    Format: [type: B][lat: i][lon: i][alt: h][yaw: H][fix: B]
    """
    unpacked = struct.unpack('!BiihHB', data)
    lat = unpacked[1] / 1_000_000.0
    lon = unpacked[2] / 1_000_000.0
    alt = unpacked[3] / 100.0
    yaw = unpacked[4] / 100.0
    fix = unpacked[5]

    fix_str = "LOCKED" if fix == 1 else "LOST"
    print(f"Received VGPS: fix={fix_str} Lat: {lat:.6f} Lon: {lon:.6f} Alt: {alt:.2f}m Yaw: {yaw:.1f}deg")

    gga = create_gga(lat, lon, alt)
    rmc = create_rmc(lat, lon)

    if ser is not None:
        try:
            ser.write((gga + "\r\n").encode())
            ser.write((rmc + "\r\n").encode())
        except Exception as e:
            print(f"Serial write error: {e}")

    print(gga)
    print(rmc)
    return

def process_tracking_data(data: bytes, mavlink: mavutil.mavserial = None):
    unpacked_payload = struct.unpack('!BIIIIII', data)
    print(f"Received Tracking: ResX: {unpacked_payload[1]} ResY: {unpacked_payload[2]} TgtX: {unpacked_payload[3]} TgtY: {unpacked_payload[4]} TgtW: {unpacked_payload[5]} TgtH: {unpacked_payload[6]}")

    if FRAME_TYPE == "QUADCOPTER":
        roll_value = 1500
        pitch_value = max(1000, min(2000, map_range(unpacked_payload[5] * unpacked_payload[6], 0, unpacked_payload[1] * unpacked_payload[2], 1000, 1500)))
        throttle_value = 1500
        yaw_value = max(1000, min(2000, map_range(unpacked_payload[3], 0, unpacked_payload[1], 1000, 2000)))
    elif FRAME_TYPE == "QUADPLANE":
        roll_value = max(1000, min(2000, map_range(unpacked_payload[3], 0, unpacked_payload[1], 1000, 2000)))
        pitch_value = 1500
        throttle_value = 1500
        yaw_value = 1500

    print(f"RC Overrides -> Roll: {roll_value} Pitch: {pitch_value} Throttle: {throttle_value} Yaw: {yaw_value}")

    if mavlink is not None:
        send_rc(
            roll = roll_value,          # CH1
            pitch = pitch_value,        # CH2
            throttle = throttle_value,  # CH3
            yaw = yaw_value,            # CH4
            m = mavlink
        )
    return

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(False)
sock.bind((UDP_IP, UDP_PORT))
print(f"Ground Handler listening on {UDP_IP}:{UDP_PORT}")

# serial_gps = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)

# serial_mavlink = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
# serial_mavlink.wait_heartbeat()
# print("MAVLink Connected")

while True:
    try:
        data, addr = sock.recvfrom(1024)
        if data[0] == ord('\x7E'):
            length = data[1]
            payload = data[2:2+length]
            crc_recv = data[2+length]
            crc_calc = crc8.dvb_s2(data[:2+length])
            if crc_recv == crc_calc:
                if payload[0] == ord('\x01'):
                    process_gps_data(payload)
                elif payload[0] == ord('\x02'):
                    process_tracking_data(payload)
                elif payload[0] == 0x10:  # New high-precision VGPS format
                    process_vgps_telemetry(payload)
                else:
                    print(f"Unknown payload type: {payload[0]}")
            else:
                print(f"Checksum mismatch. Received CRC: {crc_recv} != Calculated CRC: {crc_calc}")
        else:
            print(f"Invalid packet header: {data[0]}")
    except socket.error:
        pass
