
"""_summary_

Returns:
    _type_: _description_
"""
# pylint: disable=too-many-positional-arguments
# Import Libraries
# import configparser
import numpy as np

# # Create a ConfigParser object
# config = configparser.ConfigParser()

# # Read the config.ini file
# # Please set the path properly before usage
# config.read('./configurations/config.ini')

# LOW = config.getfloat('perturbation_variables', 'LOW')
# HIGH = config.getfloat('perturbation_variables', 'HIGH')
# STEP = config.getfloat('perturbation_variables', 'STEP')
# Now Let's write functions to provide perturbations to each column specific to each data type

# BASIC PERTURBATIONS FOR THE VERSION I: Provides Gaussian Noise by protecting
# the standard deviation and mean properties of the current column.
def categorical_perturbation(column, num_permutations=5):
    """
    Provide categorical perturbation to datasets.

    Args:
        X_Cal (pandas.DataFrame): Input DataFrame.

    Returns:
        tuple: A tuple containing lists of perturbation column names, types, 
                severities, and perturbed datasets.
    """
    column = column.copy()  # Make a copy to avoid modifying the original DataFrame
    for _ in np.arange(num_permutations):
        column_perturbed = np.random.permutation(column)  # Shuffle the values in the column
    return column_perturbed



# Assuming you have a function to generate alternative instances for numerical columns
def gaussian_perturbation(column, severity):
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
    column = column.copy()

    # Calculate mean and standard deviation of the original column
    original_mean = column.mean()
    original_std = column.std()

    # Generate perturbation based on severity
    perturbation = np.random.normal(loc=0, scale=original_std * severity, size=len(column))

    # Apply perturbation while preserving mean and standard deviation
    return original_mean + perturbation


def uniform_perturbation(column, severity):
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
    column = column.copy()

    # Calculate mean and standard deviation of the original column
    # original_mean = column.mean()
    original_range = column.max() - column.min()

    # Generate perturbation based on severity
    perturbation = np.random.uniform(low=-original_range * severity,
                                     high=original_range * severity, size=len(column))

    # Apply perturbation while preserving mean
    return column + perturbation

# pylint: disable=invalid-name, too-many-arguments
def perturb_dataset(X_cal,
                    y_cal,
                    categorical_features=None,
                    noise_type='uniform',
                    scale_factor=5,
                    severity=0.5):
    '''
    Function used to perturb the dataset for the calibration process.
    '''
    perturbed_X_cal = np.tile(X_cal.copy(), (scale_factor,1))
    scaled_X_cal = perturbed_X_cal.copy()
    scaled_y_cal = np.tile(y_cal.copy(), scale_factor)
    assert noise_type in ['uniform', 'gaussian'], \
        "Noise type must be either 'uniform' or 'gaussian'."

    for f in range(scaled_X_cal.shape[1]):
        if f in categorical_features:
            perturbed_X_cal[:,f] = categorical_perturbation(perturbed_X_cal[:,f])
        elif noise_type == 'uniform':
            # Apply numerical alternative perturbation to the selected column -- uniform
            perturbed_X_cal[:,f] = uniform_perturbation(perturbed_X_cal[:,f], severity)
        elif noise_type == 'gaussian':
            # Apply numerical alternative perturbation to the selected column -- gaussian
            perturbed_X_cal[:,f] = gaussian_perturbation(perturbed_X_cal[:,f], severity)
    return perturbed_X_cal, scaled_X_cal, scaled_y_cal, scale_factor
