import torch

class Bubble():
    _id = 0

    def __init__(self, size, location):
        self.id = Bubble._generate_ID()
        self.size = size
        self.location = location
        self.lifecycle = 0

    @classmethod
    def _generate_ID(cls):
        cls._id += 1
        return cls._id
    
    def addlifecycle(self):
        self.lifecycle += 1

    def find_intersection_gpu(self, other):
        rect1 = torch.tensor(self.location, device='cpu', dtype=torch.float32)
        rect2 = torch.tensor(other.location, device='cpu', dtype=torch.float32)

        ### extract x-y coordinates for the upper-left and bottom-right corners of each rectangle
        x1_1, y1_1 = rect1[0]
        x2_1, y2_1 = rect1[2]
        x1_2, y1_2 = rect2[0]
        x2_2, y2_2 = rect2[2]

        ### find the cooridnates of intersection rectangles
        x1_i = torch.max(x1_1, x1_2)
        y1_i = torch.max(y1_1, y1_2)
        x2_i = torch.min(x2_1, x2_2)
        y2_i = torch.min(y2_1, y2_2)

        ### calculate width and height of intersection area, if no intersection between the two, the width and height will be negative
        width = torch.max(torch.tensor(0.0, device='cpu'), x2_i - x1_i)
        height = torch.max(torch.tensor(0.0, device='cpu'), y2_i - y1_i)

        ### calculate intersection area
        area = width*height

        return area
    
    def find_intersection(self, other):
        rect1 = self.location
        rect2 = other.location

        ### extract x-y coordinates for the upper-left and bottom-right corners of each rectangle
        x1_1, y1_1 = rect1[0]
        x2_1, y2_1 = rect1[2]
        x1_2, y1_2 = rect2[0]
        x2_2, y2_2 = rect2[2]

        ### find the cooridnates of intersection rectangles
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        ### calculate width and height of intersection area, if no intersection between the two, the width and height will be negative
        width = max(0, x2_i - x1_i)
        height = max(0, y2_i - y1_i)

        ### calculate intersection area
        area = width*height

        return area





