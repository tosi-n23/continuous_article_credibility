from apex import amp
from model import ClassificationModel
# from longformer_model.model import ClassificationModel
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import argparse
import gc


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False




# Concept drift function comparing 
def concept_drift(full_GT_data, eval_df, model_type, model_name, new_model, use_cuda, num_labels):
    
    _, groundTruth_eval = train_test_split(full_GT_data, test_size=0.07, random_state=123, shuffle=True)

    GT_model = ClassificationModel(model_type=model_type, model_name=model_name, num_labels=num_labels, use_cuda=use_cuda, args={'num_train_epochs': 1, 'reprocess_input_data': True, 'overwrite_output_dir': True})
    groundTruth, _, _ = GT_model.eval_model(groundTruth_eval)

    model = ClassificationModel(model_type=model_type, model_name=new_model, num_labels=num_labels, use_cuda=use_cuda, args={'num_train_epochs': 1, 'reprocess_input_data': True, 'overwrite_output_dir': True})
    increLeraner, _, _ = model.eval_model(eval_df)
    
    if groundTruth['F1-Score'] < increLeraner['F1-Score']:
        return 1
    else:
        return 0




def continuous_learner(fullset, full_GT_data, model_type, model_name, new_model, num_labels, use_cuda, output_dir):
    
    # Increment Partitions
    increment = [0.5, 0.75, 1]
    
    for i in increment:
        fullset = fullset.sample(frac = i, random_state = 123)

        logger.info(" Incremental training on {} percent of article credibility data.".format(i*100))
        train_df, eval_df = train_test_split(fullset, test_size=0.07, random_state=123, shuffle=True)
    
        model = ClassificationModel(model_type=model_type, model_name=model_name, num_labels=num_labels, use_cuda=use_cuda, args={'num_train_epochs': 1, 'reprocess_input_data': True, 'overwrite_output_dir': True})
    
    
        # Train the model
        model.train_model(train_df = train_df, eval_df=None,  output_dir=output_dir, show_running_loss=True)
    
        logger.info("--—-Deleting training dataframe loader--—-")
        del train_df
        gc.collect()               
    


        # Monitor Concept Drift
        metric = concept_drift(full_GT_data, eval_df, model_type, model_name, new_model, use_cuda, num_labels)
        
        logger.info("--—-Deleting evaluation dataframe loader--—-")
        del eval_df
        gc.collect()
        
        print(metric)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='longformer', type=str)
    parser.add_argument('--model_name', default='groundTruth', type=str)
    parser.add_argument('--new_model', default='increModel', type=str)
    parser.add_argument('--num_labels', default=10, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--gt_datapath', default='full_GT_data', type=str)
    parser.add_argument('--datapath', default='fullset', type=str)
    parser.add_argument('--output_dir', default='longformer', type=str)
    opt = parser.parse_args()
    
    
    
    credibility_classifier = {
        'groundTruth': '/home/tosi-n/credibility_api/longformer/checkpoint-8000',
        'increModel': '/home/tosi-n/credibility_api/longformer',
    }


    dataset_files = {
            'full_GT_data': '/home/tosi-n/credibility_api/data/bal_set.csv',
            'fullset': '/home/tosi-n/credibility_api/data/bal_set.csv',
    }
    
    
    
    
    # Incremental Data
    fullset = pd.read_csv(dataset_files[opt.datapath], sep='\t')
    
    # Ground Truth Data
    full_GT_data = pd.read_csv(dataset_files[opt.gt_datapath], sep='\t')

    fullset.rename(columns={'labels': 'labels_',
                            'factors': 'labels'},
                        inplace=True)

    full_GT_data.rename(columns={'labels': 'labels_',
                            'factors': 'labels'},
                        inplace=True)



    # opt.model_class = model_classes[]
    opt.model_name = credibility_classifier[opt.model_name]
    opt.new_model = credibility_classifier[opt.new_model]
    opt.fullset_data = fullset
    opt.full_GT_data = full_GT_data 
    # model_type =  opt.model_type



    continuous_learner(opt.fullset_data, opt.full_GT_data, opt.model_type, opt.model_name, opt.new_model, opt.num_labels, opt.use_cuda, opt.output_dir)






if __name__ == '__main__':
	main()