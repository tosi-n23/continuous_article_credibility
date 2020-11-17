#!/bin/bash
################################################################################
#                                 Script                                       #
#                                                                              #
#                                                                              #
#                                                                              #
#                                                                              #
################################################################################
################################################################################
################################################################################
#                                                                              #
#                          Longformer Base Model                               #
#                                   +                                          #
#                               Label Dict                                     #
#                                                                              #
################################################################################
################################################################################
################################################################################

cd "./src/longformer_model/"

# # gcloud compute scp tosi-n@logically-dev-tensorflow:/home/tosi-n/svm_cred_api/base_model.zip /home/tosi-n/article_credibility_api/src/longformer_model/

# # gcloud compute scp tosi-n@logically-dev-tensorflow:/home/tosi-n/svm_cred_api/label_dict.pkl /home/tosi-n/article_credibility_api/src/longformer_model/

# gsutil cp -r gs://fake_news_corpus11/longformer_credibility_cls_mdl/base_model.zip ./src/longformer_model/

mkdir "./models"

gsutil cp -r gs://fake_news_corpus11/longformer_credibility_cls_mdl/label_dict.pkl /home/tosi-n/article_credibility_api-1/src/longformer_model/models

gsutil cp -r gs://fake_news_corpus11/lonformer_cred_cls/base_model/ /home/tosi-n/article_credibility_api-1/src/longformer_model/models



# # mv "./label_dict.pkl" "./models/label_dict.pkl"

# zip -d ./base_model.zip "__MACOSX/*"

# unzip "./base_model.zip"

# rm "./base_model.zip"