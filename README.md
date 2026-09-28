# bionic-limb-activity-inference
Implementation pipeline for activity inference from transradial bionic limb sensor data. Contains preprocessing, feature extraction, model training, and evaluation scripts for activity classification from accelerometer data recorded from the wrist flexor region.

# Artifact Scope & Code Structure
> **Data Availability Note:** Due to privacy and research ethics board agreements, raw experimental sensor data cannot be publicly hosted in this repository. There are two provided scripts. Clustering_Script.py contains the complete, self-contained pipeline used to process the data, extract features, evaluate clustering models, and generate plots. Transfer_Learning_Script.py contains the source code for executing the supervised learning approach of our inference analysis.

```text
.
├── Clustering_Script.py      # Standalone pipeline for clustering (preprocessing, feature extraction, clustering, evaluation)
├── Transfer_Learning_Script.py  # Pipeline for transfer learning implementation using Harnet10 model
├── requirements.txt       # Dependency specifications
└── README.md              # Documentation
```
# Quickstart Guide for Unsupervised Learning Approach

Follow these steps to set up the environment and execute the pipeline on your dataset.

## 1. Install Dependencies
Ensure you have Python 3.8+ installed, then install all required packages:
```bash
pip install -r requirements.txt
```
## 2. Prepare your dataset
Provide a .csv file containing continuous tri-axial accelerometer recordings. Your dataset must include the following column headers:
Activity: Ground truth class labels
X,Y,Z : Acceleration values along each respective axis

## 3. Configure data path
Open Clustering_Script.py in any text editor and update the csv_path variable at the top of the script with the path to your CSV file.

## 4. Run the Pipeline
Execute the script Clustering_Script.py

### Expected Output
Visualization outputs include a ground-truth distribution scatter plot alongside comparative 2D PCA projections generated across five clustering models: $k$-Means, Gaussian Mixture Models (GMM), Agglomerative Hierarchical Clustering, DBSCAN, and OPTICS.

# Quickstart Guide for Supervised Learning Approach

## 1. Install Dependencies
Ensure you have Python 3.8+ installed, then install all required packages:
```bash
pip install -r requirements.txt
```
## 2. Prepare your dataset
Provide two .csv files containing continuous tri-axial accelerometer recordings. One having your training dataset and another having your testing dataset. 

## 3. Configure data path
Open Transfer_Learning_Script.py in any text editor and update the csv_path variable at the top of the script with the path to your CSV files.

## 4. Run the Pipeline
Execute the script Transfer_Learning_Script.py


