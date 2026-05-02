import serial
import struct
import time

PORT = '/dev/ttyACM0'
TARGET_ERPM = 1000
DURATION = 5.0

def crc16_ccitt(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc

def make_packet(payload):
    n = len(payload)
    if n < 256:
        header = bytes([0x02, n])
    else:
        header = bytes([0x03, (n >> 8) & 0xFF, n & 0xFF])
    crc = crc16_ccitt(payload)
    return header + payload + bytes([(crc >> 8) & 0xFF, crc & 0xFF, 0x03])

def set_rpm(ser, rpm):
    payload = struct.pack('>Bi', 8, int(rpm))
    ser.write(make_packet(payload))

def set_current(ser, amps):
    payload = struct.pack('>Bi', 6, int(amps * 1000))
    ser.write(make_packet(payload))

def request_values(ser):
    ser.write(make_packet(bytes([4])))

def read_packet(ser, timeout=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b[0] == 0x02:
            l = ser.read(1)
            if len(l) < 1:
                return None
            length = l[0]
        elif b[0] == 0x03:
            l = ser.read(2)
            if len(l) < 2:
                return None
            length = (l[0] << 8) | l[1]
        else:
            continue
        payload = ser.read(length)
        crc_bytes = ser.read(2)
        stop = ser.read(1)
        if len(payload) != length or len(crc_bytes) != 2 or len(stop) != 1 or stop[0] != 0x03:
            return None
        crc_recv = (crc_bytes[0] << 8) | crc_bytes[1]
        if crc16_ccitt(payload) != crc_recv:
            return None
        return bytes(payload)
    return None

def parse_values(payload):
    if len(payload) < 54 or payload[0] != 4:
        return None
    return {
        'temp_fet': struct.unpack('>h', payload[1:3])[0] / 10.0,
        'temp_motor': struct.unpack('>h', payload[3:5])[0] / 10.0,
        'i_motor': struct.unpack('>i', payload[5:9])[0] / 100.0,
        'i_in': struct.unpack('>i', payload[9:13])[0] / 100.0,
        'duty': struct.unpack('>h', payload[21:23])[0] / 1000.0,
        'rpm': struct.unpack('>i', payload[23:27])[0],
        'v_in': struct.unpack('>h', payload[27:29])[0] / 10.0,
        'fault': payload[53],
    }

ser = serial.Serial(PORT, 115200, timeout=0.05)
try:
    print(f"Sending {TARGET_ERPM} ERPM for {DURATION}s")
    start = time.time()
    last_query = 0.0
    while time.time() - start < DURATION:
        set_rpm(ser, TARGET_ERPM)
        if time.time() - last_query > 0.1:
            request_values(ser)
            pkt = read_packet(ser, timeout=0.05)
            if pkt is not None:
                v = parse_values(pkt)
                if v:
                    print(f"RPM={v['rpm']:6d}  V={v['v_in']:5.2f}  Im={v['i_motor']:6.2f}  Ib={v['i_in']:6.2f}  duty={v['duty']:5.2f}  Tfet={v['temp_fet']:4.1f}  Tmot={v['temp_motor']:4.1f}  fault={v['fault']}")
            last_query = time.time()
        time.sleep(0.02)
    set_rpm(ser, 0)
    time.sleep(0.3)
    set_rpm(ser, 0)
    print("Stopped")
finally:
    set_rpm(ser, 0)
    ser.close()