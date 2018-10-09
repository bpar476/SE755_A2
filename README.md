## Dataset interface
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



