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
    def __init__(self, **kwargs):
        
        self.style_list = ['seaborn-whitegrid', 'style_sheet.mplstyle']
        self.style = self._make_plot_style()

    def _add_labels(self, ax):
        """
        Function to add labels and title to plots. 
        Called from inside the `make_x_plot` functions.
        Arguements: ax the axis plot in matplotlib.
        Returns: None
        """
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.title(self.title)
        plt.tight_layout()
        
    def _make_plot_style(self):
        """
        Modify the plotting style sheet in matplotlib. 
        
        Arguements: 
            original_style: The default style sheet in matplotlib or seaborn. 
                Default is white_background.
                
            modify_style: The new style parameters to partiall 
                overwrite the default style
        Returns: None
        """
        plt.style.use(self.style_list)
        
    def _make_annotation_format(self, y_value, **kwargs):
        """
        YOU NEED AN ERROR TO RAISE IF NO ANNOTATION FORMAT IS GIVEN
        
        Function to format annotations for plots
        
        Arguements: None
        
        Returns: label, formatted using kwargs
        """
        if 'money' in kwargs:
            label = f"${int(y_value):,}"
                
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
        
        return plt.figtext(0.48,
                           -0.02, 
                           kwargs['caption'],
                           wrap=True,
                           horizontalalignment='center', 
                           fontsize=10)
        
        
        
        
        
        