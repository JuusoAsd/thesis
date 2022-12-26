import os


class FileManager:
    def __init__(self, folder_path, output_type=None):
        self.path = folder_path
        self.files = os.listdir(self.path)
        self.files.sort()
        self.create_output = output_type

    def get_next_file(self):
        if len(self.files) > 0:
            self.iterator = iter(open(os.path.join(self.path, self.files.pop(0))))
            self.iterator.__next__()
        else:
            return None

    def get_next_event(self, start=False):
        try:
            val = next(self.iterator)
        except:
            self.get_next_file()
            val = next(self.iterator)
        if self.create_output is None:
            return val.rstrip().split(",")
        else:
            return self.create_output(val.rstrip().split(","))


class Trade:
    # 557111802,1.2373,10.0,12.373,1640044800076,false
    def __init__(self, input_list):
        self.timestamp = int(input_list[4])
        self.price = float(input_list[1])
        self.size = float(input_list[2])
