import torch

def split_sep(sequence, sep, cls, pad, max_length=128):
    chunk = []
    for val in sequence:
        if val == sep:
            yield chunk
            chunk = []
        else:
            if val not in [cls, pad]:
                chunk.append(val)
    
    if chunk:
        yield chunk
