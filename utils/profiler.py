import time


class Profiler:
    def __init__(self):
        self.record_item = {}

    def start(self, name):
        if name not in self.record_item:
            self.record_item[name] = {}
            self.record_item[name]["count"] = 0

        self.record_item[name]["start_time"] = time.time()

    def end(self, name):
        self.record_item[name]["end_time"] = time.time()
        self.record_item[name]["time"] = (
            self.record_item[name]["end_time"] - self.record_item[name]["start_time"]
        )
        if "avg_time" not in self.record_item[name]:
            self.record_item[name]["avg_time"] = self.record_item[name]["time"]
        else:
            self.record_item[name]["avg_time"] = (
                (self.record_item[name]["avg_time"] * self.record_item[name]["count"])
                + self.record_item[name]["time"]
            ) / (self.record_item[name]["count"] + 1)
        self.record_item[name]["count"] += 1

    def print(self, name):
        print(f"Avg time for {name}: {self.record_item[name]['avg_time']}s")

    def print_all(self):
        for name in self.record_item:
            self.print(name)
