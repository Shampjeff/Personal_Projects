import matplotlib.pyplot as plt
import seaborn as sns

class PlotRoutine:
    """
    A general plotting routine that builds on matplotlib and seaborn. Style is inherited. 
    
    Attributes:
        dataframe: the source of data in pandas dataframe format
        x_label: String format of the x axis label
        y_label: String format of the y axis label
        title: String format of the plot title
    Key Word Arguements:
        
        int: string, display interger values
        round_0: string, one decimal place round
        round_2: string, two decimal place round
        money: string, format USD format comma
            placements. 
        annot: bool, adds value annotation
        caption: string, text of caption
        rotate: int, degree of label rotation (Bar plot)

    """
    def __init__(self, dataframe, x_axis,
                 y_axis, x_label, y_label, title):
        
        self.data = dataframe
        self.x_axis = x_axes
        self.y_axis = y_axes
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.style = self._make_plot_style(self)

    def _add_labels(self,ax):
        """
        Function to add labels and title to plots. 
        Called from inside the `make_x_plot` functions.
        Arguements: ax the axis plot in matplotlib.
        Returns: None
        """
        ax.set_ylabel(self.y_label)
        ax.set_xlabel(self.x_label)
        ax.set_title(self.title)
        plt.tight_layout()
        
    def _make_plot_style(self,
                         original_style='white_background',
                         modify_style='style_sheet.mplstyle'):
        """
        Modify the plotting style sheet in matplotlib. 
        
        Arguements: 
            original_style: The default style sheet in matplotlib or seaborn. 
                Default is white_background.
                
            modify_style: The new style parameters to partiall 
                overwrite the default style
        Returns: None
        """
        plt.use.style([original_style, modify_style])
        
    def _make_annotation_format(self, **kwargs):
        """
        Function to format annotations for plots
        
        Arguements: None
        
        Returns: label, formatted using kwargs
        """
        if 'int' in kwargs:
            label = f"{float(y_value):.0f}"
            if 'money' in kwargs:
                label = f"${int(label):,}"

        if 'round_0' in kwargs:
            label = f"{y_value:.1f}"
            if 'money' in kwargs:
                label = f"${float(label):,}"

        if 'round_2' in kwargs:
            label = f"{y_value:.2f}"
            if 'money' in kwargs:
                label = f"${float(label):,}"
        return label
    
    def _make_caption(self, **kwargs):
        """
        Function to add caption to a plot figure
        Arguements: None
        Key Word Agruement:
            caption: sting, the caption to display
        Returns: matplotlib command for the caption
        
        MIGHT NOT NEED THE RETURN LINE!!!!
        """
        
        return plt.figtext(0.5,
                           -0.05, 
                           kwargs['caption'],
                           wrap=True,
                           horizontalalignment='center', 
                           fontsize=14)
        
        
        
        
        
        