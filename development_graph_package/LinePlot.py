import matplotlib.pyplot as plt
import seaborn as sns
from GeneralPlot import PlotRoutine

class LinePlot(PlotRoutine):
    """
    Line plot for bivariate data with ability 
    to plot categoricals hues. 
    
    Atttributes: 
        make_line_plot, display line plot 
        with options for caption and annotation.
        
    """
    
    def __init__(self, x_axis, y_axis,
                 x_label, y_label, title,**kwargs):
        
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        
        PlotRoutine.__init__(self, **kwargs)
        
    
    def make_line_plot(self, hue=None, annot_labels=None, 
                       annot_values=None, annot_x_value=None, **kwargs):
        """
        Function to make an x,y line plot from a pandas dataframe.
        
        Arguments: 
            hue: string, categorical fields associated with x,y data.
            annot_labels: list of string, categorical labels to display
            annot_values: list of floats, categorical values to display
        
        Returns: None
        """
        plt.figure(figsize=(8,4))
        ax = sns.lineplot(x=self.x_axis,
                          y=self.y_axis,
                          hue=hue)
      
        if annot_labels != None:
            for i in range(len(annot_labels)):
                ax.annotate(
                s=f"{annot_labels[i].title()}: {self._make_annotation_format(annot_values[i],**kwargs)}", 
                    xy=(annot_x_value, annot_values[i]), 
                    fontsize=10,
                    xytext=(10,3),         
                    textcoords="offset points"
                            )
                
        
        
        if 'caption' in kwargs:
            self._make_caption(**kwargs)
            
        plt.legend(loc='upper left')
        self._add_labels(ax)
        
