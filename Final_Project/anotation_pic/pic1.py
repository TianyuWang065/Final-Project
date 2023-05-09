import matplotlib.pyplot as plt
import numpy as np
class DrawPic:

    @staticmethod
    def draw(x,y):
        
        plt.scatter(x, y)
        plt.show()

    @staticmethod
    def draw_histogram(title,x_value,label):
        plt.title(title)
        plt.grid(ls="--",alpha=0.5)
        cm = plt.bar(label,x_value)
        for rect in cm:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2.-0.08, 1.03*height, '%s' % float(height), size=10, family="Times new roman")

        # plt.xticks(rotation=45)
        plt.show()






