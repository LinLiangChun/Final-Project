# ADL Final Project - Group 19

### Environment and Packages
- Create a virtual environment (recommended).
  ```bash
  # Create
  python -m venv venv

  # Activate
  source venv/Scripts/activate
  ```
- Intall the required packages.
  ```bash
  pip install -r requirements.txt
  ```

### Datasets
- Set up the necessary datasets.
  ```bash
  python setup_data.py
  ```

### Implement
- Implement with two tasks.
  ```bash
  python main.py --bench_name "classification_public" --output_path <path_to_save_csv>
  python main.py --bench_name "sql_generation_public" --output_path <path_to_save_csv>
  ```
