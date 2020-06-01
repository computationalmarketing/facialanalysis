
## Running the code

To run this code, you will need python3 installed. 

Step 1: Install current versions of all required packages using command `pip3 install -r requirements.txt`

Step 2: Unarchive `data.zip` file in this directory.

Step 3: Run command `python3 ./study_1_results.py` from inside this directory to replicate Study 1 results from the paper.

## Notes

Full cross-validation as implemented in the code takes days to run when using a GPU. Folder `./results` contains computation output of the cross-validation in case you want to directly construct plots based on it. (Simply comment out `for rep in tqdm(range(n_reps)):` loop and the saving of the output that follows. You also need to unzip `./results/patch_importance.json.zip`).

We also provide `requirements_full.txt` - it represents the output of `pip3 list > requirements_full.txt` on the GPU machine where the core cross-validation results were obtained (it contains more packages than required) and is provided for completeness.

Within `./data/data.csv`, `randomID` variable contains randomly generated unique identifier of a respondent in the data. There are 3 or fewer images -- and thus observations -- per respondent.

`q_to_full_name_dict` in `study_1_results.py` contains variable label definitions.

See the paper for further details.


Copyright (C) 2020 Yegor Tkachenko, Kamel Jedidi
