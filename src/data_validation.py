import pandas as pd
import great_expectations as ge

def validate_data(csv_path):
    df = pd.read_csv(csv_path)
    ge_df = ge.from_pandas(df)

    ge_df.expect_column_values_to_not_be_null("Amount")
    ge_df.expect_column_values_to_be_between("Amount", min_value=0)
    ge_df.expect_column_values_to_be_in_set("Class", [0, 1])
    ge_df.expect_table_row_count_to_be_between(min_value=1000)

    result = ge_df.validate()
    return result["success"]

if __name__ == "__main__":
    success = validate_data("data/raw/creditcard.csv")
    if not success:
        raise ValueError("❌ Data validation failed")
    print("✅ Data validation passed")
