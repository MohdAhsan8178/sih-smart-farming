"""
edge/lora.py
------------
SX1278 LoRa Packet Transport Framing, CRC16-CCITT, and Telemetry Serialization.

Implements:
  1. Pure CRC16-CCITT calculation and validation (polynomial 0x1021, init 0xFFFF).
  2. Binary packet framing with header synchronization, node addressing, sequence tracking,
     payload length guards, and tail CRC checksums.
  3. Compact binary telemetry payload serialization and deserialization (16 bytes packed).
  4. Mockable SX1278Driver interface decoupled from physical Linux SPI bus.

Packet Wire Format (Header: 8 bytes, Payload: N bytes, Checksum: 2 bytes):
  [0]     Sync Byte (0xAA)
  [1]     Protocol Version (0x01)
  [2..3]  Source Node ID (uint16 big-endian, 0..65535)
  [4]     Message Type (uint8: 0x01=TELEMETRY, 0x02=COMMAND, 0x03=ACK, 0x04=ALERT)
  [5..6]  Sequence Number (uint16 big-endian, 0..65535)
  [7]     Payload Length N (uint8, 0..240)
  [8..8+N-1] Payload Data (N bytes)
  [8+N..8+N+1] CRC16-CCITT (uint16 big-endian over bytes [0..8+N-1])
"""

import struct
from typing import Dict, Any, Optional, Tuple, Union


# ==============================================================================
# Protocol Constants
# ==============================================================================
LORA_SYNC_BYTE: int = 0xAA
LORA_PROTOCOL_VERSION: int = 0x01
MAX_PAYLOAD_LEN: int = 240
HEADER_LEN: int = 8
CRC_LEN: int = 2

# Message types
MSG_TELEMETRY: int = 0x01
MSG_COMMAND: int = 0x02
MSG_ACK: int = 0x03
MSG_ALERT: int = 0x04

VALID_MSG_TYPES = {
    MSG_TELEMETRY: "TELEMETRY",
    MSG_COMMAND: "COMMAND",
    MSG_ACK: "ACK",
    MSG_ALERT: "ALERT"
}


# ==============================================================================
# Pure CRC16-CCITT (Polynomial 0x1021, Init 0xFFFF)
# ==============================================================================

def crc16_ccitt(data: bytes, init_val: int = 0xFFFF) -> int:
    """
    Compute CRC16-CCITT checksum over bytes.
    Standard: Polynomial 0x1021 (x^16 + x^12 + x^5 + 1), Init 0xFFFF.

    Test vector:
      b"123456789" -> 0x29B1
    """
    crc = int(init_val) & 0xFFFF
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


# ==============================================================================
# Packet Framing & Deserialization
# ==============================================================================

def build_lora_packet(node_id: int,
                      msg_type: int,
                      seq_num: int,
                      payload: bytes) -> bytes:
    """
    Assemble and serialize a complete LoRa frame with CRC16-CCITT.

    Parameters:
      node_id: uint16 source node identifier (0..65535).
      msg_type: uint8 message type (0x01..0x04).
      seq_num: uint16 packet sequence number (0..65535).
      payload: Raw binary payload bytes (0..240 bytes).

    Returns:
      Raw packet bytes including 8-byte header, N-byte payload, and 2-byte CRC.
    """
    if node_id < 0 or node_id > 65535:
        raise ValueError(f"node_id {node_id} out of uint16 range [0, 65535].")
    if msg_type not in VALID_MSG_TYPES:
        raise ValueError(f"Unknown msg_type {msg_type}. Valid types: {list(VALID_MSG_TYPES.keys())}")
    if seq_num < 0 or seq_num > 65535:
        raise ValueError(f"seq_num {seq_num} out of uint16 range [0, 65535].")
    if len(payload) > MAX_PAYLOAD_LEN:
        raise ValueError(f"Payload length {len(payload)} exceeds max allowable {MAX_PAYLOAD_LEN} bytes.")

    header = struct.pack(
        ">BBHBHB",
        LORA_SYNC_BYTE,
        LORA_PROTOCOL_VERSION,
        node_id,
        msg_type,
        seq_num,
        len(payload)
    )

    frame_body = header + payload
    checksum = crc16_ccitt(frame_body)
    tail = struct.pack(">H", checksum)

    return frame_body + tail


def parse_lora_packet(packet_bytes: bytes) -> Dict[str, Any]:
    """
    Parse, validate, and unpack an incoming LoRa packet.

    Checks:
      1. Minimum packet length (>= 10 bytes: 8 header + 2 CRC).
      2. Sync byte matches 0xAA.
      3. Protocol version matches 0x01.
      4. Message type is known.
      5. Frame length matches declared payload length + 10.
      6. CRC16-CCITT integrity matches.

    Returns:
      Dict with parsed fields and validation status:
        valid: bool
        status: 'ok', 'CRC_MISMATCH', 'INVALID_SYNC', 'LENGTH_MISMATCH', etc.
        node_id: int
        msg_type: int
        msg_type_name: str
        seq_num: int
        payload: bytes
    """
    if len(packet_bytes) < (HEADER_LEN + CRC_LEN):
        return {
            "valid": False,
            "status": "PACKET_TOO_SHORT",
            "detail": f"Received {len(packet_bytes)} bytes, minimum valid packet is {HEADER_LEN + CRC_LEN} bytes."
        }

    # Verify Sync Byte
    sync = packet_bytes[0]
    if sync != LORA_SYNC_BYTE:
        return {
            "valid": False,
            "status": "INVALID_SYNC",
            "detail": f"Expected sync byte 0x{LORA_SYNC_BYTE:02X}, got 0x{sync:02X}."
        }

    version = packet_bytes[1]
    if version != LORA_PROTOCOL_VERSION:
        return {
            "valid": False,
            "status": "UNSUPPORTED_VERSION",
            "detail": f"Expected version {LORA_PROTOCOL_VERSION}, got {version}."
        }

    node_id, msg_type, seq_num, payload_len = struct.unpack(">HBHB", packet_bytes[2:8])

    expected_total_len = HEADER_LEN + payload_len + CRC_LEN
    if len(packet_bytes) != expected_total_len:
        return {
            "valid": False,
            "status": "LENGTH_MISMATCH",
            "detail": f"Packet length {len(packet_bytes)} does not match declared payload ({payload_len} + 10)."
        }

    # Verify Checksum
    frame_body = packet_bytes[:-2]
    received_crc = struct.unpack(">H", packet_bytes[-2:])[0]
    computed_crc = crc16_ccitt(frame_body)

    if received_crc != computed_crc:
        return {
            "valid": False,
            "status": "CRC_MISMATCH",
            "detail": f"CRC error: computed 0x{computed_crc:04X} != received 0x{received_crc:04X}.",
            "computed_crc": computed_crc,
            "received_crc": received_crc
        }

    payload = packet_bytes[HEADER_LEN:HEADER_LEN + payload_len]

    return {
        "valid": True,
        "status": "ok",
        "node_id": node_id,
        "msg_type": msg_type,
        "msg_type_name": VALID_MSG_TYPES.get(msg_type, "UNKNOWN"),
        "seq_num": seq_num,
        "payload_len": payload_len,
        "payload": payload,
        "crc": received_crc
    }


# ==============================================================================
# Compact Telemetry Binary Serialization (16 Bytes)
# ==============================================================================

def encode_telemetry_payload(moisture_pct: float,
                             soil_temp_c: float,
                             canopy_temp_c: float,
                             battery_mv: int,
                             flow_rate_lpm: float,
                             cumulative_liters: float,
                             fault_flags: int = 0) -> bytes:
    """
    Pack telemetry readings into a compact 16-byte fixed binary structure:
      Format: '>HHhHHHI' (16 bytes)
        moisture: uint16 (centi-percent: 0..10000 -> 0.00%..100.00%)
        soil_temp: int16 (centi-deg C: -5000..10000 -> -50.00C..100.00C)
        canopy_temp: int16 (centi-deg C: -5000..10000 -> -50.00C..100.00C)
        battery_mv: uint16 (millivolts: 0..65535 mV)
        flow_rate: uint16 (centi-LPM: 0..65535 -> 0.00..655.35 LPM)
        fault_flags: uint16 (bitmask)
        cumulative_liters: uint32 (centi-liters: 0..4294967295 -> 0.00..42,949,672.95 L)
    """
    m_scaled = max(0, min(10000, int(round(moisture_pct * 100.0))))
    st_scaled = max(-5000, min(10000, int(round(soil_temp_c * 100.0))))
    ct_scaled = max(-5000, min(10000, int(round(canopy_temp_c * 100.0))))
    bat = max(0, min(65535, int(battery_mv)))
    flow_scaled = max(0, min(65535, int(round(flow_rate_lpm * 100.0))))
    faults = max(0, min(65535, int(fault_flags)))
    vol_scaled = max(0, min(4294967295, int(round(cumulative_liters * 100.0))))

    return struct.pack(">HhhHHHI", m_scaled, st_scaled, ct_scaled, bat, flow_scaled, faults, vol_scaled)


def decode_telemetry_payload(payload_bytes: bytes) -> Dict[str, Any]:
    """
    Unpack a 16-byte binary telemetry payload into floating point measurements.
    """
    if len(payload_bytes) != 16:
        raise ValueError(f"Telemetry payload must be exactly 16 bytes, got {len(payload_bytes)} bytes.")

    m_s, st_s, ct_s, bat, flow_s, faults, vol_s = struct.unpack(">HhhHHHI", payload_bytes)

    return {
        "moisture_pct": float(round(m_s / 100.0, 2)),
        "soil_temp_c": float(round(st_s / 100.0, 2)),
        "canopy_temp_c": float(round(ct_s / 100.0, 2)),
        "battery_mv": int(bat),
        "flow_rate_lpm": float(round(flow_s / 100.0, 2)),
        "fault_flags": int(faults),
        "cumulative_liters": float(round(vol_s / 100.0, 2))
    }


# ==============================================================================
# Mockable Driver Class
# ==============================================================================

class SX1278Driver:
    """
    Mockable SX1278 transceiver driver.
    Decoupled from physical spidev and GPIO interrupts for testing and simulation.
    """
    def __init__(self, node_id: int = 1, spi_bus: Optional[Any] = None):
        self.node_id = int(node_id)
        self.spi_bus = spi_bus
        self.seq_num: int = 0
        self.tx_buffer: List[bytes] = []
        self.rx_queue: List[bytes] = []

    def send(self, msg_type: int, payload: bytes) -> bytes:
        """Frame and transmit packet (appends to tx_buffer in mock mode)."""
        packet = build_lora_packet(self.node_id, msg_type, self.seq_num, payload)
        self.seq_num = (self.seq_num + 1) % 65536
        self.tx_buffer.append(packet)
        return packet

    def queue_mock_rx(self, packet: bytes) -> None:
        """Inject a packet into the receiver queue (used for testing)."""
        self.rx_queue.append(packet)

    def receive(self) -> Optional[Dict[str, Any]]:
        """Pop and parse the next packet from the rx queue."""
        if not self.rx_queue:
            return None
        raw_pkt = self.rx_queue.pop(0)
        return parse_lora_packet(raw_pkt)
