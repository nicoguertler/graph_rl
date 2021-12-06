class ArrowConfig:
    def __init__(self, scale, width, head_size, color):
        self.scale = scale
        self.width = width
        self.head_size = head_size
        self.color = color

def get_default_subgoal_colors():
    matplotlib_default = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    rgb_colors = [tuple(int(h[i:i+2], 16)/255. for i in [1, 3, 5]) for h in matplotlib_default]
    return rgb_colors
