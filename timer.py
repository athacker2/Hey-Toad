import time
import threading
import random

class Timer_Manager(threading.Thread):
    timers = dict()

    def __init__(self):
        threading.Thread.__init__(self)
    
    def create_timer(name, timer_time):
        if(name == "default"):
            name = "timer{0}".format(random.randint(0, 10000))
        
        Timer_Manager.timers[name] = Timer(timer_time, name)
        print("created timer: {0}".format(Timer_Manager.timers[name].name))
    
    def check_timers():
        for timer in list(Timer_Manager.timers.values()):
            if(timer.expired()):
                print("Timer {0} expired.".format(timer.name))
                Timer_Manager.timers.pop(timer.name)
    
    def run(self):
        while True:
            # print("Running Timer Manager Thread")
            if(len(Timer_Manager.timers) > 0):
                Timer_Manager.check_timers()
            time.sleep(0) # yield

class Timer:
    def __init__(self, timer_time, name):
        self.start_time = time.time()
        self.run_time = timer_time
        self.name = name
    
    def expired(self):
        if (self.start_time + self.run_time) - time.time() < 0:
            return True
        return False