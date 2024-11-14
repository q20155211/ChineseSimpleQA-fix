import pandas as pd
import os


if __name__ == "__main__":
  input_directory = 'evaluation/models_results'
  output_file = 'evaluation/leaderboard.csv'

  results = []
  for filename in os.listdir(input_directory):
      if filename.endswith(".csv") and "simpleqa_" in filename:
          model_name = os.path.splitext(filename)[0].replace("simpleqa_", "")
          file_path = os.path.join(input_directory, filename)
          
          df = pd.read_csv(file_path)

          for accuracy_type in ['correct', 'incorrect', 'not_attempted', 'given_attempted_accuracy', 'F1']:
              row_data = {'model': model_name, 'accuracy_type': accuracy_type}
              
              temp_df = df[['primary_category', accuracy_type]].copy()
              temp_df = temp_df.rename(columns={accuracy_type: 'accuracy'})
              
              for _, row in temp_df.iterrows():
                  primary_category = row['primary_category']
                  value = row['accuracy']*100
                  
                  col_name = primary_category
                  row_data[col_name] = value
              
              results.append(row_data)

  final_df = pd.DataFrame(results)
  final_df.to_csv(output_file, index=False)
