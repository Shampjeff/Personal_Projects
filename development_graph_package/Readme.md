# Development work for data visualization package
MVP for a data visualization python package that builds on matplotlib and seaborn for a simplier way to call common plots. 
Current support for Bar, histogram, and Line plots only.
The files handle all the formatting style, annotations, captions, and labeling and allow the user to simply 
input the values from a pandas dataframe. 

## `GeneralPlot.py`
General plotting class (PlotRoutine) that handles the caption, labels, and annotation formatting for specific plots. 
PlotRoutine is the parent class for LinePlot and BarPlot.

## `BarPlot.py`
Class that can be called for producing bar plots and histogram with functionallity inherited from PlotRoutine.

**required inputs:**  

pandas series for x and y values, x and y labels, and title

**Methods:** 

`make_bar_plot` - optional key word agruements for caption, annotations, and annotation formatting.

`make_hist_plot` - optional key word agruements for caption, annotations, and annotation formatting. Ignores y value inputs.

## `LinePlot.py`
Class that can be call for producing line plots with availablity to handle to multiple categorical lines. 

**required inputs:**

pandas series for x and y values, x and y labels, and title

**Methods:** 

`make_line_plot` - optional arguements for categorical hue, categorical annotations with corresponding values, and key word
arguements for formatting and caption. 
