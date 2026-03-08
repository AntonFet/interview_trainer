import math
import struct

def rms(data):
    count = len(data) // 2
    shorts = struct.unpack_from("<" + "h" * count, data)
    sum_squares = sum(sample * sample for sample in shorts)

    return math.sqrt(sum_squares / count) if count > 0 else 0