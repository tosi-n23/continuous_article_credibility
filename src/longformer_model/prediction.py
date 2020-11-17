from .model import ClassificationModel
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import numpy as np
import pickle
import pandas as pd
import logging
import argparse
import gc


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

base_long = './src/longformer_model/models/base_model'

long_model = ClassificationModel('roberta', base_long, num_labels=10, use_cuda=True, args={'reprocess_input_data': True, 'overwrite_output_dir': True})

label_dict = pickle.load(open('./src/longformer_model/models/label_dict.pkl', 'rb'))

def predict(content):
	content = [content]
	preds_l, outputs = long_model.predict(content)

	confidence = []
	probabilities = np.array([softmax(element) for element in outputs])
	for i in probabilities:
		long_probs = np.amax(i)
		confidence.append(long_probs)

	for k, v in label_dict.items():
		if preds_l[0] == k:
			i = v
			# print(v)
			# return i

	return i, str(confidence[0])

# j = predict('A new study shows people who are voracious #selfie posters on sites like Facebook and Instagram tend to have higher levels of narcissism and psychopathic behaviors than folks to opt to stick to posting pics of their dinner or family pets.\r\n\r\nA new year, a new opportunity to see obnoxious content on your social media feeds. You had an amazing time in the Caribbean over the holidays and want to post some tanned, shirtless selfies? Fine. You proposed underneath the Eiffel Tower on New Year\'s Eve and are sharing pics of the ring on Twitter? Yeesh, get it over with already. In fact, you might be tempted to swear off all those social braggarts as part of your New Year\'s resolution. Now, there\'s good reason to want to do so.\r\n\r\n— Trevor Henry (@RiotQuickshot) January 7, 2015\r\n\r\nA new study, from The Ohio State University shows that men who post more online photos of themselves than others score higher on measures of narcissism and psychopathy on personality tests. Perhaps unsurprisingly, men who say they edit their selfies before they post them score even higher on measures of narcissism and self-objectification — an indication of how much they put priority on their appearance.\r\n\r\n"It\'s not surprising that men who post a lot of selfies and spend more time editing them are more narcissistic, but this is the first time it has actually been confirmed in a study," said Jesse Fox, the lead author of the study, and a professor of communication at OSU.\r\n\r\nAnti-Social, But Still Normal\r\n\r\nAs much as you might like to gleefully insult your selfie-loving pals in the comments section under their self portraits, you should be aware the men in the study all scored within the "normal" range of behavior. They simply had higher-than-average levels of anti-social personality traits.\r\n\r\nJust a quick reminder of what some of these terms mean: narcissism is a belief that you are smarter, better and more attractive than other people — with some underlying insecurity present. Psychopathy, on the other hand, is marked by a lack of empathy and regard for others with a tendency to behave impulsively.\r\n\r\n— Joey Ryan (@JoeyRyanOnline) January 5, 2015\r\n\r\nThe data were gathered by surveying 800 men about their photo-posting behavior on social media. Interestingly, posting more photos was linked to both narcissism and psychopathy, but psychopathic traits were not connected to people who edit their photos before spamming them onto our newsfeeds. That\'s because psychopathic traits are characterized by impulsivity — wanting to get those #selfies on the Web for all to see and glorify as soon as is humanly possible.\r\n\r\nEditing Leads To Self-Objectification\r\n\r\nBy contrast, those who obsessively crop, apply filters and otherwise edit their snaps are more likely to engage in self-objectification, which can result in a laundry list of negative behaviors, such as valuing your appearance over other positive traits. Self-objectification is linked in women to depression and eating disorders, but has rarely been studied in heterosexual men. That\'s why studies like these are important.\r\n\r\n— Geraldo Rivera (@GeraldoRivera) January 4, 2015\r\n\r\nIf all that doesn\'t convince you to lay off putting yourself on a pedestal, consider this: the researchers conclude a self-reinforcing cycle comes into play when it comes to self-objectification. People who self-objectify tend to post more selfies, leading to more feedback online, leading to — you guessed it — a tendency to post more photos of oneself online.\r\n\r\nSo as you enter a new year, filled with hope and promise, why not make things easier on yourself and others by sticking to posting pics of your cat? Mr. Mittens — and you — will thank you for it.')

# print(j)
