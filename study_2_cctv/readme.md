
## Running the code

To run this code, you will need python3 installed. 

Step 1: Install current versions of all required packages using command `pip3 install -r requirements.txt`

Step 2: Download `data.zip` password-protected file via provided link and unzip it in this directory (to get the link to the data and the password, you need to complete the form: [link](https://app.getparampara.com/s/b4323991-fbb1-4e84-9bab-b6bf4cc054c7)).

Step 3: Run command `python3 ./study_2_results.py ALL` from inside this directory to replicate Study 2 results from the paper.

## Notes

Full cross-validation as implemented in the code takes days to run when using a GPU. Folder `./results_face` contains computation output of the cross-validation in case you want to directly construct plots based on it. (Simply comment out `for rep in tqdm(range(n_reps)):` loop and the saving of the output that follows. You also need to unzip `./results_face/patch_importance.json.zip`).

We also provide `requirements_full.txt` - it represents the output of `pip3 list > requirements_full.txt` on the GPU machine where the core cross-validation results were obtained (it contains more packages than required) and is provided for completeness.

See the paper for further details.


Copyright (C) 2020 Yegor Tkachenko, Kamel Jedidi
