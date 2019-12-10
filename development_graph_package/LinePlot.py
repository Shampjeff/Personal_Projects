import matplotlib.pyplot as plt
import seaborn as sns
from .GeneralPlot import PlotRoutine

class LinePlot(PlotRoutine):
    """
    Line plot for bivariate data with ability 
    to plot categoricals hues. 
    
    Atttributes: 
        make_line_plot, display line plot 
        with options for caption and annotation.
        
    """
    
    def __init__(self, *args, **kwargs):
        
        PlotRoutine.__init__(self, dataframe, x_axis,
                             y_axis, x_label, y_label, title):
        
    
    def make_line_plot(self, hue=None, annot_labels=None, 
                       annot_values=None, *args, **kwargs):
        """
        Function to make an x,y line plot from a pandas dataframe.
        
        Arguments: 
            hue: string, categorical fields associated with x,y data.
            annot_labels: list of string, categorical labels to display
            annot_values: list of floats, categorical values to display
        
        Returns: None
        """
        plt.figure(figsize=(12,8))
        ax = sns.lineplot(x=self.x_axis,
                          y=self.y_axis,
                          hue=hue)
      
        if annot_labels != None:
            for i in range(len(annot_labels)):
                ax.annotate(
                    s=f"{annot_labels[i].title()}: {annot_values[i]:.1f}", 
                    xy=(x_value, annot_values[i]), 
                    fontsize=14,
                    xytext=(10,5),         
                    textcoords="offset points")

        _make_annotation_format(self, **kwargs)
        
        if 'caption' in kwargs:
            _make_caption(self, **kwargs)
            
        plt.legend(loc='upper left')
        _add_labels(self, ax)
        
