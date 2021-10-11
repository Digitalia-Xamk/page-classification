# page-classification
Python scripts for separating empty, handwritten, typed or mixed pages containing text from jpg files
Uses scikit learn package for solving classification problem. Models are formed calculating some statistics from tesseract output. 
Models are essentially based on the tesseracts confidence score. 
User must first run AI-KA-mode.py such that jpg files are placed under folder from which the program is started. The class names are defined by the name of the corresponding folder in the end of the path. 
Then one have a csv file, where all the pages are discribed such that one line represents a single page. 
Also the location (full path) of the file is stored. Once one have a csv file, models for actual testing is formed by script AI-result.py. These models can be tested using script Model-test.py. 
For more information contact tuomo.raisanen@xamk.fi
