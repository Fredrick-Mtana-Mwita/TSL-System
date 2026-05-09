from collections import deque

# keep the last N frames (sequence length)

SEQUENCE_LENGTH = 30

class SequenceBuffer:
    def __init__(self):
        self.buffer = deque(maxlen = SEQUENCE_LENGTH)

    def add(self, landmarks):
        self.buffer.append(landmarks)

    def is_ready(self):
        return len(self.buffer) == SEQUENCE_LENGTH
    
    def get_sequence(self):
        return list(self.buffer)
    
    def clear(self):
        self.buffer.clear()


# Global buffer (for demo , later per-user)
sequence_buffer = SequenceBuffer()