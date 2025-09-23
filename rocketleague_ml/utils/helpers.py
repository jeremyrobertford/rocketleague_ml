def convert_byte_to_float(bytes: int):
    byte_val = max(0, min(255, bytes))
    return (byte_val - 128) / 127.0
