import pandas as pd
import itertools
from math import log2
import os
import re

# Global Data Storage
main_dataframe = None
probability_results = {}
header_combinations = {}
combination_probabilities = {}


class CSVPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def _parse_range(self, data_range):
        match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", data_range)
        if not match:
            raise ValueError("Invalid range format. Use 'A1:D10'.")
        start_col, start_row, end_col, end_row = match.groups()
        return (int(start_row) - 1, int(end_row)), (
            self._excel_col_to_index(start_col), self._excel_col_to_index(end_col) + 1)

    @staticmethod
    def _excel_col_to_index(col):
        return sum((ord(char.upper()) - ord('A') + 1) * 26 ** i for i, char in enumerate(reversed(col))) - 1

    def load_and_process_csv(self, data_range=None):
        try:
            df = pd.read_csv(self.file_path, header=None if data_range else 0)
            if data_range:
                row_slice, col_slice = self._parse_range(data_range)
                df = df.iloc[row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]]
                df.columns = df.iloc[0]
                df = df.drop(df.index[0])
            global main_dataframe
            main_dataframe = df
            return df
        except Exception as e:
            return str(e)


def calculate_entropy(probabilities):
    return -sum(p * log2(p) for p in probabilities if p > 0)


def calculate_maximum_entropy(combination_prob_df):
    num_unique_combinations = len(combination_prob_df)
    if num_unique_combinations > 0:
        return log2(num_unique_combinations)
    else:
        return 0  # or some appropriate value for distributions with no variations


def calculate_information_measure(combination, combination_prob_df):
    total_combinations = combination_prob_df['Count'].sum()
    information_measures = []
    for _, row in combination_prob_df.iterrows():
        joint_prob = row['Probability']
        individual_probs_product = 1
        for column in combination:
            value_index = probability_results[column][column].index(str(row[column]))
            individual_prob = probability_results[column]['Probability'][value_index]
            individual_probs_product *= individual_prob
        info_measure = log2(joint_prob / individual_probs_product) if individual_probs_product > 0 else float('inf')
        information_measures.append(info_measure)
    combination_prob_df['InformationMeasure'] = information_measures
    return combination_prob_df


def calculate_significance(combination, combination_prob_df):
    order = len(combination)
    significance_values = []

    for _, row in combination_prob_df.iterrows():
        joint_prob = row['Probability']
        chi_squared_by_2N = row['ChiSquaredBy2N']

        if order == 2:
            # For order 2, significance is equal to chi-squared
            significance = chi_squared_by_2N
        else:
            # For orders greater than 2, calculate significance using entropy
            shannon_entropy = row['ShannonEntropy']
            max_entropy = row['MaximumEntropy']
            if joint_prob > 0 and max_entropy > 0:
                exponent = pow(shannon_entropy / max_entropy, order * 0.5)
                significance = (1 / joint_prob) * pow(chi_squared_by_2N, exponent)
            else:
                significance = None  # Handle undefined cases

        significance_values.append(significance)

    combination_prob_df['Significance'] = significance_values
    return combination_prob_df


class CombinationProbability:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def calculate_combination_probability(self, combination):
        sorted_df = self.data_frame.sort_values(list(combination)).reset_index(drop=True)
        combination_counts = sorted_df.groupby(list(combination)).size().reset_index(name='Count')
        combination_counts['Probability'] = combination_counts['Count'] / len(self.data_frame)
        return combination_counts

    def calculate_joint_probability(self, combination):
        sorted_df = self.data_frame.sort_values(list(combination)).reset_index(drop=True)
        combination_counts = sorted_df.groupby(list(combination)).size().reset_index(name='Count')
        combination_counts['Probability'] = combination_counts['Count'] / combination_counts['Count'].sum()
        return combination_counts

    def calculate_chi_squared_by_2N(self, combination, combination_prob_df):
        N = len(self.data_frame)  # Total number of observations
        chi_squared_values = []

        for _, row in combination_prob_df.iterrows():
            joint_prob = row['Probability']
            individual_probs_product = 1
            for column in combination:
                value_index = probability_results[column][column].index(str(row[column]))
                individual_prob = probability_results[column]['Probability'][value_index]
                individual_probs_product *= individual_prob

            # Calculate chi-squared value
            if individual_probs_product > 0:
                chi_squared_by_2N = ((joint_prob - individual_probs_product) ** 2) / (2 * individual_probs_product)
            else:
                chi_squared_by_2N = float('inf')  # Handle division by zero

            chi_squared_values.append(chi_squared_by_2N)

        combination_prob_df['ChiSquaredBy2N'] = chi_squared_values
        return combination_prob_df

    def calculate_all_combinations_probabilities(self, support_threshold):
        combination_probabilities.clear()
        headers = self.data_frame.columns.tolist()
        max_order = min(6, len(headers))

        for order in range(2, max_order + 1):
            for comb in itertools.combinations(headers, order):
                comb_prob_df = self.calculate_joint_probability(comb)

                # Filter combinations based on support threshold
                comb_prob_df = comb_prob_df[comb_prob_df['Probability'] >= support_threshold]

                if comb_prob_df.empty:
                    continue

                # Calculate information measure
                comb_prob_df = calculate_information_measure(comb, comb_prob_df)
                comb_prob_df = comb_prob_df[comb_prob_df['InformationMeasure'] >= 0]

                if comb_prob_df.empty:
                    continue

                if order > 2:
                    comb_prob_df['ShannonEntropy'] = [calculate_entropy(comb_prob_df['Probability'].tolist())] * len(
                        comb_prob_df)
                    max_entropy = calculate_maximum_entropy(comb_prob_df)
                    comb_prob_df['MaximumEntropy'] = [max_entropy] * len(comb_prob_df)

                comb_prob_df = self.calculate_chi_squared_by_2N(comb, comb_prob_df)
                comb_prob_df = calculate_significance(comb, comb_prob_df)

                # Calculate MI - Chi_sqr_2N
                comb_prob_df['MI - Chi_sqr_2N'] = comb_prob_df['InformationMeasure'] - comb_prob_df['Significance']

                combination_probabilities[comb] = comb_prob_df.to_dict('records')


class HeaderCombinations:
    def __init__(self):
        if main_dataframe is None:
            raise ValueError("No data frame loaded. Please load the data frame before generating combinations.")

        self.headers = main_dataframe.columns.tolist()

    def generate_combinations(self, order):
        # Directly return combinations for the specific order
        return list(itertools.combinations(self.headers, order))


def get_probability(column_name):
    # Retrieve the probability data for the given column from Data.py
    return probability_results.get(column_name, f"No probability data found for column {column_name}.")


class ColumnProbability:
    def __init__(self, data_frame):
        global main_dataframe, probability_results
        self.data_frame = data_frame
        main_dataframe = data_frame
        self.calculate_all_probabilities()

    def calculate_probability(self, column_name):
        counts = self.data_frame[column_name].value_counts(dropna=False)
        total_counts = counts.sum()
        probabilities = counts / total_counts

        probability_df = pd.DataFrame({
            column_name: counts.index,
            'Count': counts.values,
            'Probability': probabilities.values
        })

        # Sort the DataFrame by the column values
        probability_df = probability_df.sort_values(by=column_name).reset_index(drop=True)

        # Save the results for later retrieval in Data.py
        probability_results[column_name] = probability_df.to_dict('list')

    def calculate_all_probabilities(self):
        for column in self.data_frame.columns:
            self.calculate_probability(column)


def reset_data_module():
    global main_dataframe, probability_results, header_combinations, combination_probabilities
    main_dataframe = None
    probability_results = {}
    header_combinations = {}
    combination_probabilities = {}


def store_header_combinations():
    global header_combinations
    if main_dataframe is None:
        print("No data loaded to generate header combinations.")
        return
    header_combinations_obj = HeaderCombinations()
    max_order = min(6, len(main_dataframe.columns))
    for order in range(2, max_order + 1):
        header_combinations[order] = header_combinations_obj.generate_combinations(order)


def display_combinations(order):
    if order in header_combinations:
        print(f"Order {order} combinations: {header_combinations[order]}")
    else:
        print(f"No combinations found for order {order}.")


def display_combinations_up_to(order):
    for o in range(2, order + 1):
        display_combinations(o)


def calculate_and_store_combination_probabilities():
    comb_prob = CombinationProbability(main_dataframe)
    for order in header_combinations:
        for headers in header_combinations[order]:
            # Calculate probabilities for the current combination
            prob_df = comb_prob.calculate_combination_probability(headers)
            # Augment with information measure and chi-squared values
            prob_df_with_info = calculate_information_measure(headers, prob_df)
            prob_df_with_chi_squared_by_2N = comb_prob.calculate_chi_squared_by_2N(headers, prob_df_with_info)
            # Store the augmented result in a dictionary
            combination_probabilities[headers] = prob_df_with_chi_squared_by_2N.to_dict('list')


def export_to_csv():
    # Get user input for headers
    user_input = input("Enter headers to export (separated by commas): ")
    selected_headers = [header.strip() for header in user_input.split(',')]

    # Get user input for maximum order
    max_order_input = input("Enter the maximum order to export (2-6): ")
    try:
        max_order = int(max_order_input)
        if max_order < 2 or max_order > 6:
            print("Invalid order. Please enter a number between 2 and 6.")
            return
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return

    # Check if the combination_probabilities dictionary is empty
    if not combination_probabilities:
        print("No combination probabilities to export.")
        return

    # Create an empty list to store DataFrames
    dfs = []

    # Process each combination
    for combination, prob_dict in combination_probabilities.items():
        if any(header in combination for header in selected_headers) and len(combination) <= max_order:
            temp_df = pd.DataFrame(prob_dict)
            temp_df['Combination'] = ', '.join(combination)  # Add combination as a column
            # Check if the significance was calculated (i.e., is not None)
            temp_df = temp_df[temp_df['Significance'].notnull()]
            if not temp_df.empty:
                dfs.append(temp_df)  # Append DataFrame to list

    # Concatenate all DataFrames in the list
    export_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


    # Define static columns that are always expected
    expected_cols = ['Count', 'Probability', 'InformationMeasure', 'ChiSquaredBy2N', 'Significance', 'MI - Chi_sqr_2N']
  
    # Add ShannonEntropy and MaximumEntropy to expected_cols only if they exist in export_df
    if 'ShannonEntropy' in export_df.columns:
        expected_cols.insert(2, 'ShannonEntropy')
    if 'MaximumEntropy' in export_df.columns:
        expected_cols.insert(3, 'MaximumEntropy')
  
    # Define dynamic columns
    dynamic_cols = [col for col in export_df.columns if col not in expected_cols and col != 'Combination']
  
    # Define the order of columns for export
    ordered_cols = ['Combination'] + dynamic_cols + expected_cols
  
    # Reorder the columns if export_df is not empty
    if not export_df.empty:
        export_df = export_df[ordered_cols]

    # Check if the DataFrame is not empty
    if export_df.empty:
        print("No data found for the specified criteria.")
        return

    # Create the export filename
    headers_str = "_".join(selected_headers)
    export_filename = f"{headers_str}_uptoOrder{max_order}.csv"

    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    export_dir = os.path.join(current_dir, 'Exported')

    # Check and create the 'Exported' directory if it does not exist
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # Construct the file path
    filepath = os.path.join(export_dir, export_filename)

    # Save the dataframe to CSV
    export_df.to_csv(filepath, index=False)
    print(f"Exported data to {filepath}")


def main():
    while True:
      read_me_response = input("Have you read the 'ReadMe' file yet?\n(Yes/No): ").strip().lower()
      if read_me_response == "yes":
          break
      elif read_me_response == "no":
          help_response = input("Do you need help locating the 'ReadMe' file?\n(Yes/No): ").strip().lower()
          if help_response == "yes":
              print("\nIF you are running this file in 'Replit', check that under the Name of this program, "
                    "there is an option '< >'. Click this to see all the files available to you and read "
                    "the files from there.\nIF you are running it using an IDE, it should be located in the "
                    "folder where you ran this program from.\n")
  
    reset_data_module()
    file_path = input("Enter the path to your CSV file: ")
    if not os.path.exists(file_path):
        print("File does not exist. Please check the path.")
        return

    data_range = input("Enter the desired data range (e.g., 'A1:D10'): ")

    preprocessor = CSVPreprocessor(file_path)
    data_frame = preprocessor.load_and_process_csv(data_range)

    if isinstance(data_frame, str):
        print(data_frame)
        return
    else:
        print("DataFrame loaded and processed successfully.")
        print("Calculation in Progress ...")
        column_prob = ColumnProbability(data_frame)
        column_prob.calculate_all_probabilities()

        # Prompt user for support threshold
        print(f"Choose a proper support threshold. Such as 0.02 or 0.05 or 0.5 etc.")
        support_input = input("Enter the support threshold (default 0): ")
        try:
            support_threshold = float(support_input)
            print(f"support threshold is set to: {support_threshold}")
        except ValueError:
            support_threshold = 0
            print(f"Invalid input. Defaulting to threshold: 0.")

        # Calculate combination probabilities
        comb_prob = CombinationProbability(data_frame)
        comb_prob.calculate_all_combinations_probabilities(support_threshold)

        print("Calculating information measures and chi-squared values for combinations...")
        calculate_and_store_combination_probabilities()

        store_header_combinations()  # Store header combinations for each order
        print("Calculation Complete.")

    while True:
        print("Command: ('show', 'table', 'header', 'viewdata', 'perms',' export', 'quit')")
        command = input("Enter a command: ").lower()

        if command == 'quit':
            break
        elif command == 'export':
            export_to_csv()
        elif command == 'viewdata':
            print("Current Data.main_dataframe:")
            if main_dataframe is not None:
                print(main_dataframe)
            else:
                print("No main dataframe loaded.")

            print("\nCurrent Data.probability_results:")
            print(probability_results)

            print("\nCurrent Data.header_combinations:")
            print(header_combinations)

            print("\nCurrent Data.combination_probabilities:")
            for comb, prob in combination_probabilities.items():
                print(f"Combination: {comb}, Probability Table: {prob}")
        elif command == 'show':
            column_name = input("Enter the column name to show probability: ")
            prob_data = probability_results.get(column_name)
            if prob_data:
                print(pd.DataFrame(prob_data))
            else:
                print(f"No probability data found for column {column_name}.")
        elif command == 'perms':
            order_input = input(
                "Enter the order for header combinations (e.g., '5' for up to order 5, '15' for only order 5): ")
            try:
                order = int(order_input)
                if len(order_input) == 1:  # Display a range of orders up to the specified order
                    display_combinations_up_to(order)
                elif len(order_input) == 2 and order_input.startswith('1'):  # Display a specific order
                    display_combinations(int(order_input[1]))
                else:
                    print(
                        "Invalid order. Please enter a single digit for a range or '1' followed by a digit for a specific order.")
            except ValueError as e:
                print(f"Invalid input: {e}")
        elif command == 'table':
            if main_dataframe is not None:
                print(main_dataframe)
            else:
                print("No data loaded.")
        elif command == 'header':
            if main_dataframe is not None:
                print(main_dataframe.columns.tolist())
            else:
                print("No data loaded.")
        else:
            print("Invalid command. Please enter ('show', 'table', 'header', 'viewdata', 'perms',' export', 'quit').")


# Execute the main function if this is the main module
if __name__ == "__main__":
    main()
