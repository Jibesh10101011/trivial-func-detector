1. Install required libraries
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Navigate to `utility-filter-pipeline`

    ```bash
    cd utility-filter-pipeline
    ```

3. Model development workflow

    1. Extract feature from `analysis-with-code.json`
        
        ```bash
        python3 feature_extraction.py analysis-with-code.json output/features.csv
        ```

        You will get a file at `output/features.csv` which contains all the information can findout from the `analysis-with-code.json` e.g. `loc`,`cyclomatic_complexity`,`num_statements`,`fan_in`,`fan_out`,`num_called_unique` etc


    2. Generate Label from the `output/features.csv` 

        ```bash
        python3 generate_labels.py output/features.csv output/labeling.csv
        ```
    
    3. Train ML Model

        ```bash
        python3 train_model.py output/features.csv output/labeling.csv model/model.joblib
        ```
    
    4. Find the output of `analysis-with-code.json`

        ```bash
        python3 analyze.py analysis-with-code.json output/results.json
        ```
    
    5. Ranked Nodes

        ```bash
        python3 score_all.py output/features.csv model/model.joblib output/ranked_nodes.json
        ```

    6. Model testing on new dataset

        ```bash
        python3 predict.py model/model.joblib testing/new_analysis.json testing/output/predictions.csv
        ```