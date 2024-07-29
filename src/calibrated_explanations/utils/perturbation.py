
"""_summary_

Returns:
    _type_: _description_
"""
# Import Libraries
import configparser
import pandas as pd
import numpy as np

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the config.ini file
# Please set the path properly before usage
config.read('./configurations/config.ini')

LOW = config.getfloat('perturbation_variables', 'LOW')
HIGH = config.getfloat('perturbation_variables', 'HIGH')
STEP = config.getfloat('perturbation_variables', 'STEP')
# Now Let's write functions to provide perturbations to each column specific to each data type

# BASIC PERTURBATIONS FOR THE VERSION I: Provides Gaussian Noise by protecting
# the standard deviation and mean properties of the current column.
def categorical_perturbation(df, column_name, num_permutations=5):
    """
    Provide categorical perturbation to datasets.

    Args:
        X_Cal (pandas.DataFrame): Input DataFrame.

    Returns:
        tuple: A tuple containing lists of perturbation column names, types, 
                severities, and perturbed datasets.
    """
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    for _ in np.arange(num_permutations):
        df[column_name] = np.random.permutation(df[column_name])  # Shuffle the values in the column
    return df



# Assuming you have a function to generate counterfactual instances for numerical columns
def numerical_counterfactual_perturbation_gaussian(data, column_name, severity):
    """
    Apply Gaussian noise as numerical perturbation to a column in a DataFrame.

    Args:
        data (pandas.DataFrame): Input DataFrame.
        column_name (str): Name of the column to perturb.
        severity (float): Severity of the perturbation.

    Returns:
        pandas.DataFrame: DataFrame with perturbed column.
    """
    # Get the column to perturb
    column = data[column_name]

    # Calculate mean and standard deviation of the original column
    original_mean = column.mean()
    original_std = column.std()

    # Generate perturbation based on severity
    perturbation = np.random.normal(loc=0, scale=original_std * severity, size=len(column))

    # Apply perturbation while preserving mean and standard deviation
    perturbed_column = original_mean + perturbation

    # Replace the original column with the perturbed column
    data[column_name] = perturbed_column

    return data


def numerical_counterfactual_perturbation_uniform(data, column_name, severity):
    """
    Apply uniform noise as numerical perturbation to a column in a DataFrame.

    Args:
        data (pandas.DataFrame): Input DataFrame.
        column_name (str): Name of the column to perturb.
        severity (float): Severity of the perturbation.

    Returns:
        pandas.DataFrame: DataFrame with perturbed column.
    """
    # Get the column to perturb
    column = data[column_name]

    # Calculate mean and standard deviation of the original column
    # original_mean = column.mean()
    original_range = column.max() - column.min()

    # Generate perturbation based on severity
    perturbation = np.random.uniform(low=-original_range * severity,
                                     high=original_range * severity, size=len(column))

    # Apply perturbation while preserving mean
    perturbed_column = column + perturbation

    # Replace the original column with the perturbed column
    data[column_name] = perturbed_column

    return data


# A function to produce perturbed datasets and stores them in lists
def provide_categorical_perturbation_to_data_sets(X_Cal): # pylint: disable=invalid-name
    # Iterate over columns and provide perturbation one by one, store all data frames in a list
    """
    Provide categorical perturbation to datasets.

    Args:
        X_Cal (pandas.DataFrame): Input DataFrame.

    Returns:
        tuple: A tuple containing lists of perturbation column names, types, 
                severities, and perturbed datasets.
    """
    list_of_perturb_column = []
    list_of_perturb_type = []
    list_of_perturb_severity = []
    list_of_df = []

    for column_name, dtype in X_Cal.dtypes.items():

        if dtype == 'category':
            for severity in np.arange(LOW,HIGH,STEP):
                copy_x_cal = X_Cal.copy()
                # for each column we take a clean dataset and perturb only one column
                perturbed_x_cal= categorical_perturbation(copy_x_cal, column_name, severity )
                # let's store the new dataset
                list_of_df.append(perturbed_x_cal)
                list_of_perturb_column.append(column_name)
                list_of_perturb_type.append("permutation")
                list_of_perturb_severity.append(severity)

    return list_of_perturb_column, list_of_perturb_type, list_of_perturb_severity, list_of_df

# pylint: disable=invalid-name, too-many-arguments
def provide_numerical_perturbation_to_datasets(X_Cal, list_of_df, list_of_perturb_column,
                                               list_of_perturb_type, list_of_perturb_severity,
                                               noise_type):
    """
    Provide numerical perturbation to datasets.

    Args:
        X_Cal (pandas.DataFrame): Input DataFrame.
        list_of_df (list): List of perturbed datasets.
        list_of_perturb_column (list): List of perturbed column names.
        list_of_perturb_type (list): List of perturbation types.
        list_of_perturb_severity (list): List of perturbation severities.
        noise_type (str): Type of noise for perturbation.

    Returns:
        tuple: A tuple containing lists of perturbation column names, types, 
                severities, and perturbed datasets.
    """
    # Assuming you have a DataFrame named X_Cal
    # Loop through each column and its data type in X_Cal
    assert noise_type in ['uniform', 'gaussian'], \
        "Noise type must be either 'uniform' or 'gaussian'."

    for column_name, dtype in X_Cal.dtypes.items():

        # Check if the data type is float
        if dtype in ('float64','int64'): # check those also in other OS

            # Loop through perturbation severities (from 0.1 to 0.5 for example)
            for severity in np.arange(LOW, HIGH, STEP):

                # Create a copy of the original dataset
                copy_x_cal = X_Cal.copy()
                if noise_type == 'uniform':
                    # Apply numerical counterfactual perturbation to the selected column -- uniform
                    perturbed_x_cal = numerical_counterfactual_perturbation_uniform(
                                                            copy_x_cal, column_name, severity)
                elif noise_type == 'gaussian':
                    # Apply numerical counterfactual perturbation to the selected column -- gaussian
                    perturbed_x_cal = numerical_counterfactual_perturbation_gaussian(
                                                            copy_x_cal, column_name, severity)
                else:
                    perturbed_x_cal = None # Should not reach here

                # Store the perturbed dataset in a list
                list_of_df.append(perturbed_x_cal)

                # Store information about the perturbation
                list_of_perturb_column.append(column_name)
                list_of_perturb_type.append("counterfactual")
                list_of_perturb_severity.append(severity)

    return list_of_perturb_column, list_of_perturb_type, list_of_perturb_severity, list_of_df

def get_perturbed_datasets_summary(list_of_perturb_column, list_of_perturb_type,
                                   list_of_perturb_severity, list_of_df):
    """
    Summarize perturbed datasets.

    Args:
        list_of_perturb_column (list): List of perturbed column names.
        list_of_perturb_type (list): List of perturbation types.
        list_of_perturb_severity (list): List of perturbation severities.
        list_of_df (list): List of perturbed datasets.

    Returns:
        pandas.DataFrame: DataFrame summarizing the perturbed datasets.
    """
    # Create a DataFrame using the lists
    perturbed_df = pd.DataFrame({
        'Perturbed_Data': list_of_df,
        'Perturbed_Column': list_of_perturb_column,
        'Perturbation_Type': list_of_perturb_type,
        'Perturbation_Severity': list_of_perturb_severity
    })
    return perturbed_df
