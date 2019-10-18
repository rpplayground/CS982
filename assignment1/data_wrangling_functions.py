import re as re
import math

def trim_year_column_names(data_frame):
    # Grab the column names from the dataframe.
    columns_to_process = data_frame.columns
    column_renaming_plan = {}
    new_column_names = []
    for column_name in columns_to_process:
        # If the column begins with 4 numbers, assume it is a year column
        if re.search("[0-9]{4}", column_name):
            new_column_name = column_name[0:4]
            # Create a dictionary consisting of entries specifying current_column_name : new_column_name :
            column_renaming_plan.update({column_name : new_column_name})
            # Also build a list of the new column names as this may be useful
            new_column_names = new_column_names + [new_column_name]
    # Now appply the new names to the columns
    new_data_frame = data_frame.rename(columns = column_renaming_plan)
    # Return the dataframe
    return new_data_frame, new_column_names

def custom_round(x, up=True):
    if up:
        power = math.ceil(math.log10(x))
    else:
        power = math.floor(math.log10(x))
    value =  math.pow(10, power)
    return power, value

def find_min_and_max (data_frame, column_name):
       
    min_power, min_value = custom_round(data_frame[column_name].min(), up=False)
    max_power, max_value = custom_round(data_frame[column_name].max(), up=True)
    
    return min_power, min_value, max_power, max_value

def add_annotation(ax, row, x_column, y_column, label_column, offset_x, offset_y):
    offset = 50
    ax.annotate(\
        s = row[label_column],\
        xy = (row[x_column], row[y_column]),\
        xytext = (offset_x * offset, offset_y * offset),\
        bbox = dict(boxstyle="round", fc="0.8"),\
        arrowprops = dict(width = 2),\
        textcoords = 'offset pixels')

def look_up_offsets(offset_label):
    offset_lookup = dict(\
        OC = (0,1),\
        TR = (1,1),\
        QP = (1,0),\
        BR = (1,-1),\
        HP = (0,-1),\
        BL = (-1, -1),\
        QT = (-1,0),\
        TL = (-1,1))
    offset_tuple = offset_lookup[offset_label]
    return offset_tuple[0], offset_tuple[1]

def plot_points_of_interest(data_frame, points_of_interest, x_column, y_column, size_column, label_column, ax):
    for point_of_interest in points_of_interest:
        offset_x, offset_y = look_up_offsets(point_of_interest[1])
        row = data_frame.loc[data_frame[label_column] == point_of_interest[0]]
        add_annotation(ax, row, x_column, y_column, label_column, offset_x, offset_y)


def label_max_and_mins(data_frame, x_column, y_column, size_column, label_column, ax):
    trimmed_data_frame = data_frame.loc[:, [x_column, y_column, size_column, label_column]].dropna()
    list_of_columns = [x_column, y_column, size_column]
    for column in list_of_columns:
        max_x = trimmed_data_frame.loc[trimmed_data_frame[column].idxmax()]
        add_annotation(ax, max_x, x_column, y_column, label_column, -1, -1)
        min_x = trimmed_data_frame.loc[trimmed_data_frame[column].idxmin()]
        add_annotation(ax, min_x, x_column, y_column, label_column, 1, 1)
