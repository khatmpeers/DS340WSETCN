# SE-TCN for DS 340W

This repository contains the code and data needed to reproduce the results from my (Lalith Pranav Polakam) presentation and procedure. Please follow the directions below
to get the repository up and running without issue. 

## Main Files

The files in this repository can be generally sorted into four "classes" of function or purpose:
- Data (.csv)
- Scripts (.py, .sh, etc.)
- Logs (.log)
- Documentation (.md)

However, Data and Scripts are the more important files present.

### Data

The data is not directly stored in the repository due to the size of the directory, but the data (both raw and processed/partitioned) can be found [here](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/lqp5348_psu_edu/IgCHQfHJlpGSQ66py_I2WLCmAfDRosxC6oSmETjOwGYOSFI?e=j8CXev). Additionally, the raw data can be officially requested from the following website: https://www.research-collection.ethz.ch/entities/researchdata/61ac2f6e-2ca9-4229-8242-aed3b0c0d47c.

### Scripts

There are broadly two kinds of scripts found here. The first are helper scripts, which were used to perform various supplementary tasks needed to complete the procedure. Unless you wish to start the procedure from scratch (i.e., processing the raw data), most of these scripts can be safely ignored. 

The remaining scripts form the critical actions of this procedure. This includes the scripts that directly aided in generating and evaluating the models. More information about them can be found below in the **Procedure** section.

### Misc.

The Logs and Documentation are less important for the procedure itself. Documentation, such as this *README.md,* mainly describes aspects of the project and other useful information that could assist users, as documentation typically does. While I did not include a large amount of documentation in this project, I believe the files I did include should be useful for navigating this repository without issue. On the other hand, the log files are mainly from my runs of the procedure. They're useful in case you wish to peruse my results for yourself, but serve little purpose outside that. 

## The Procedure

This section explains how to set up this repository as a working project and carry out the procedure in a new environment. This will notably include splitting the raw data into valid partitions and training/testing the model on the variations I explored with this procedure. Below is a list of steps you can follow in order to achieve the same results.

#### Hello

### Prerequisite Technologies

Please ensure that the environment has the following technologies.
- git
- Python (3.10)

Additionally, `pyenv` is useful to have as well for easy Python version control, but it is not strictly necessary if you can directly obtain Python v3.10.

### 1. Clone the Repository

Enter a terminal or shell application in your preferred location or navigate there manually and paste the following command.

```
git clone https://github.com/khatmpeers/DS340WSETCN.git
cd DS340WSETCN
```

### 2. Create the Python Environment (venv) in the repository

This project specifically depends on Python v3.10. Verify that you have the correct Python version installed before continuing. If the current version isn't 3.10.X, then ensure that you create the environment with a v3.10.X interpreter.

MacOS/Linux: 
```
python3.10 -m venv venv
// or if using pyenv
(pyenv install 3.10) // if not installed
pyenv local 3.10
python -m venv venv
```

Windows:
```
py -3.10 -m venv venv
```

### 3. Source the venv

MacOS/Linux:
```
source venv/bin/activate
```

Windows:
```
venv\Scripts\activate
```

### 4. Install Dependencies

```
pip install -r requirements.txt
```

### 5. Download the dataset

Download the set titled "reduced_data" from:
- [My OneDrive folder](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/lqp5348_psu_edu/IgCHQfHJlpGSQ66py_I2WLCmAfDRosxC6oSmETjOwGYOSFI?e=j8CXev)

This is the partitioned and reduced dataset I used for my project. Relocate the "reduced_data" folder to the project repo. The commands below assume that the folder is in the same directory as `train.py` in the repository, but the scripts are written such that the path to the "reduced_data" and other inputs/outputs can be customized as script parameters. 

### 6. Train Surrogate XGBoost Model



```
python helper_scripts/train_xgboost.py \
  --data_path reduced_data \
  --date_column time_unix \
  --max_rows_per_trip 2000 \
  --shap_sample_size 5000 \
  --output_dir outputs_xgboost
```

### . Prepare dataset variants

Use `sparsity_splitter.py` and `feature_splitter.py` to prepare the dataset variants. 

Produce Top 5 Predictors Dataset
```
python helper_scripts/feature_splitter.py \
  --input_root reduced_data \
  --output_root reduced_data_top5_predictors \
  --shap_csv outputs_xgboost/shap_feature_importance.csv \
  --top_k 5
```

Produce Top 8 Predictors Dataset
```
python helper_scripts/feature_splitter.py \
  --input_root reduced_data \
  --output_root reduced_data_top8_predictors \
  --shap_csv outputs_xgboost/shap_feature_importance.csv \
  --top_k 5
```

Produce 2x Sparsity Dataset
```
python helper_scripts/sparsity_splitter.py \
  --input_root reduced_data \
  --output_root reduced_data_2x_sparsity \
  --step 2
```

Produce 5x Sparsity Dataset
```
python helper_scripts/sparsity_splitter.py \
  --input_root reduced_data \
  --output_root reduced_data_2x_sparsity \
  --step 5
```


### 6. Run the baseline SE-TCN experiment



