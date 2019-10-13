import re as re

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
