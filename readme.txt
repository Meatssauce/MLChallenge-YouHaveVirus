I initially tried manual cleanning of data and trained it on xgboost. However, the results were not ideal and it was difficult to properly extract key info from natural language in the data.
So I used OpenAI's GPT-2, which has state-of-the-art natural language processing capabilities.

I trained the model on the training data converted into plain text with HTML tags representing each column. 
The model's predictions then are merged with the user IDs in the input file to produce the final submission.

I used HTML tags to represent the columns as they are often used to partision plain text into different entities.
GPT-2 is trained on 8 million web pages. So it should have a somewhat decent understanding of how to read HTML data.

The scripts for GPT-2 provided by OpenAI are difficult to integrate into a separate project. So I resorted running it via commandline interface.

Step 1:
open cmd and enter
python prepare_for_gpt.py

Step 2:
open cmd, cd into the folder transformer_openai
run command python train.py --dataset infected --desc infected --submit --trainfile COVID-chat-trainingdata.csv --testfile COVID-chat-testdata.csv --n_iter 3

Step 3:
open cmd and enter
python finalise.py --is_final_estimator

Note: 
1. submission file is dataset/generated/submission/predicton.csv
2. to use different train and/or test data, replace contents of folder 'dataset'

credit:
GPT-2, OpenAI
https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde