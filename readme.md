#User Guideline
This project parse each FB post to Google API, saved the first 10 results and compute cosine similarity between FB post and google results to see whether a post is a quotation. Then we use the quotation feature in the pipeline

The idea is to build a feature vector that represent content originality and test whether this feature is predictive to depression 

step 1:
QuoteDetector2.py: parse each post to Google API and save result as matrix (run longjob with run_quote)

step 2: 
merge_doc.py: merge all the google search results to one file

step 3:
label_quoations.py: automatic annotation of quotes (run longjob with labeling.sh)

step 4: 
quotation_fea.py: generate quotation feature to be used in the pipeline

step 5:
QuoteNDepStsts.py: This script obtains basic statistics for lyrics and quotes, regression and LDA model
get_noun_trunk_lda, change define which component in the sentence you want to use in LDA
