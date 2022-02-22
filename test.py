import queue
import time

import pandas as pd


def sleep10():
    print("In Sleep 10")
    time.sleep(10)


def sleep20():
    print("In Sleep 20")
    time.sleep(20)


def sleep30():
    print("In Sleep 30")
    time.sleep(30)


if __name__ == '__main__':
    # car_sales_1
    max_heap = queue.PriorityQueue()
    max_heap.put(1)
    max_heap.put(2)
    max_heap.put(3)
    print(max_heap.get())
    print(max_heap.get())
