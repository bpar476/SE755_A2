# Usage
Clone this repository and download the required anaconda packages
Start in the `src` directory

To run a file, use the command
`python [file] [options]`

Where file is the name of the file to run the analysis with, such as occupancy_clustering.py,
and options are the command line args.

### Runnable Files
landsat_classification.py - Logistic regression model
landsat_deep_learning.py - Deep learning model
occupancy_clustering.py - K-means and GMM
traffic_bayesian_regression.py - Bayesian Ridge Regression
### Args
`--test TEST_FILE`  path to additional features to test against. Path must be
                   relative to the current directory. If supplied, results of
                   predictions against this test data will be the last thing
                   printed by this script (optional)
`--re-train`        will re-train the model and document cross validation
process
 `--analysis`        will run the model against test data and report
                   performance
 `--test-dev`        Will do a train-test and print the results to the command
                   line using the dataset from Canvas. For development
                   purpose only
                   
### Train
`--re-train`
Overrides the existing model with a new one
Inputs: none
Outputs
 - Model file saved in `/model` dir
 - CV results saved in `/cv_results` dir
### Test 
`--test TEST_FILE`
Makes predictions on new unseen data
Inputs: file with new data (may or may not have target column?)
Outputs: file with predictions in `/results` dir

### Performance
 `--analysis`
Conducts performance analysis on the trained model
Inputs: none
Outputs
    - Print accuracy/error/variance to console
    - File with untuned and tuned performance scores
    - Performance results in `/performance` dir
