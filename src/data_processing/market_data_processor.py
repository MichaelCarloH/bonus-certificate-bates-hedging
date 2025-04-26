class MarketDataProcessor:
    def __init__(self, data_folder, stock_price):
        self.data_folder = data_folder
        self.stock_price = stock_price  # Parameterized stock price
        self.data_by_maturity = {}
        self.combined_data = None

    def load_and_process_data(self):
        import os
        import pandas as pd

        # Iterate through all CSV files in the data folder
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith('.csv'):
                # Extract maturity from the file name (assumes MMDDYY format)
                maturity = self._extract_maturity_from_filename(file_name)

                # Load the CSV file into a DataFrame
                file_path = os.path.join(self.data_folder, file_name)
                df = pd.read_csv(file_path)

                # Add the Maturity column
                df['Maturity'] = maturity

                # Add a new column 'Mid_Price' based on the condition
                df['Mid_Price'] = df.apply(
                    lambda row: (row['Bid_Call'] + row['Ask_Call']) / 2 if row['Strike'] > self.stock_price else (row['Bid_Put'] + row['Ask_Put']) / 2,
                    axis=1
                )

                # Store the DataFrame by maturity
                self.data_by_maturity[maturity] = df

        # Combine all data into a single DataFrame
        self.combined_data = pd.concat(self.data_by_maturity.values(), ignore_index=True)

    def _extract_maturity_from_filename(self, file_name):
        # Extract MMDDYY from the file name and convert to a readable format
        month = file_name[:2]
        day = file_name[2:4]
        year = '20' + file_name[4:6]  # Assumes 21st century
        return f"{month}/{day}/{year}"

    def get_data_by_maturity(self):
        return self.data_by_maturity

    def get_dataframes_list(self):
        """
        Returns a list of DataFrames, each corresponding to a maturity.
        """
        return list(self.data_by_maturity.values())

# Example usage:
# processor = MarketDataProcessor("../data/marketDataClose25-04", 81.25)
# processor.load_and_process_data()
# data_by_maturity = processor.get_data_by_maturity()