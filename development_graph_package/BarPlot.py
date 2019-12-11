import matplotlib.pyplot as plt
import seaborn as sns
from GeneralPlot import PlotRoutine

class BarPlot(PlotRoutine):
    """
    Bar plot for categorical data. 
    
    Arguements: 
        dataframe: dataframe filtered to only columns to plot
        x_axis: string, x_axis categories for plot 
        y_axis: string, y_axis values for plot
        x_label: string, label for x axis
        y_label: string, label for y axis
        title: string, Plot title
     Attributes:
        make_bar_plot: Displays bar plot of sorted values (ascending) 
        make_hist_plot: Display a histogram or univariate data
    """
    def __init__(self, dataframe, x_axis,
                 y_axis, x_label, y_label, 
                 title,**kwargs):
        
        self.data = dataframe
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        # do I define or re-define the args and kwargs 
        #as self.attributes? Will that make them easier to access??
        
        PlotRoutine.__init__(self, **kwargs)
    
    def _add_bar_annotation(self, ax, **kwargs):
        """
        Function to add bar plot annotations. 
        Agruements: 
            ax: axis plot in seaborn of matplotlib
        
        Returns: None
        """
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            va = 'bottom'
            spacing=2
            
            label = self._make_annotation_format(y_value, **kwargs)
            
            ax.annotate(
                label,                      
                (x_value, y_value),         
                xytext=(0, spacing),          
                textcoords="offset points", 
                ha='center',                
                va=va, 
                fontsize=14)    
        
    def make_bar_plot(self,hue=None, **kwargs):
        """
        Function to produce bar plot from two dataframe columns. 
        
        Arguements: 
            hue: separate bars for subcategory. This is
            the same command is seaborn or matplotlib hue call. 
        
        Returns: None
        
        """
        df = self.data.sort_values(self.y_axis)
        plt.figure(figsize=(6,4))
        ax = sns.barplot(x=self.x_axis,
                         y=self.y_axis,
                         hue=hue,
                         data=df,
                         ci=None)
        if 'annot' in kwargs:
            self._add_bar_annotation(ax, **kwargs)
        if 'caption' in kwargs:    
            self._make_caption(**kwargs)
        if 'rotate' in kwargs:
            ax.tick_params(axis= 'x',
                           labelrotation= kwargs['rotate'])

        self._add_labels(ax)
        
    def make_hist_plot(self, **kwargs):
        """
        Function to display histogram of univariate data from 
        pandas dataframe. 
        
        Argurments: None
        
        Returns: None
        """
        plt.figure(figsize=(6,4))
        ax = plt.hist(self.data)
        if 'annot' in kwargs:
            self._add_bar_annotation(ax, **kwargs)
        if 'caption' in kwargs:    
            self._make_caption(**kwargs)  
        self._add_labels(ax)
        
    
                