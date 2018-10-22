# Usage
Clone this repository and download the required anaconda packages
Start in the `src` directory

To run a file, use the command
`python [filename] [options]`

Where filename is the name of the file to run the analysis with, such as occupancy_clustering.py,
and options are the command line args.

### Args
 --test TEST_FILE  path to additional features to test against. Path must be
                   relative to the current directory. If supplied, results of
                   predictions against this test data will be the last thing
                   printed by this script (optional)
 --re-train        will re-train the model and document cross validation
                   process
 --analysis        will run the model against test data and report
                   performance
 --test-dev        Will do a train-test and print the results to the command
                   line using the dataset from Canvas. For development
                   purpose only

### Train
Overrides the existing model with a new one
Inputs: none
Outputs
 - Model file saved in `/model` dir
 - CV results saved in `/cv_results` dir
### Test 
Makes predictions on new unseen data
Inputs: file with new data (may or may not have target column?)
Outputs: file with predictions in `/results` dir

### Performance
Conducts performance analysis on the trained model
Inputs: none
Outputs
    - Print accuracy/error/variance to console as usual
    - File with untuned and tuned performance scores (score measures TBC)
    - Performance results in `/performance` dir



