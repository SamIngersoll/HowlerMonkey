from container import Container
from datetime import datetime, timedelta
import time

if __name__ == '__main__':
    container = Container(learning_rate = 0.2, max_steps = 10, hidden1 = 10, hidden2 = 20, batch_size = 100, lookback=4, generation_number=0)
