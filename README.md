ARTICLE CREDIBILITY API
================
SEP 3, 2020

### What is in this repository?

  - [Credibility Classifier Prediction API Version 2](/src/api)
  - [DistilRoBERTa Base
    Model](/scripts/model_dir.sh) script for pulling model from gcp bucket to codebase
  - Continuous Learner

### File Paths

![Directory](/data/filepath.png)

### Getting Started


#### Credibility Classifier Prediction API

###### ENVIRONMENT INFORMATION

    - transformers_version: 3.0.0
    - framework: PyTorch
    - framework_version: 1.5.0
    - python_version: 3.6.9
    - system: Linux
    - ubuntu version: Ubuntu 16.04.4 LTS
    - cpu: x86_64
    - cuda version: 10.2
    - architecture: 64bit
    - fp16: False
    - use_multiprocessing: True
    - use_gpu: True
    - num_gpus: 1
    - gpu: Tesla P100-PCIE-16GB

###### Setup

With VENV Create a new virtual environment and install packages.

    virtualenv -p python3.6 venv
    source ./venv/bin/activate

Install requirements

    pip install requirements.txt

###### Edit and run script to import base\_model in required datapath

    sh ./scripts/model_dir.sh

model is trained with Longformer - global attention - Huggingface Please
run it on port 8008

###### Run Flask API

    python3.6 ./src/run.py

###### API ENDPOINT

Using a client side, run a post request using json body on port 8008

    http://127.0.0.1:8008/v2/predict/distilroberta_cls
    
    POST - 'Content-Type: application/json'

###### Request

Version 2

    { 
        "_id" : "1", 
        "title" : "Your Friday Briefing", 
        "sentiment" : { "negative": { "confidence": 0.9960937499035479 }, "neutral": { "confidence": 0.0019531445243914027 }, "positive": { "confidence": 1.9531265169053625e-8 } }, 
        "source_cred" : "high", 
        "body" : "Devils Advocate Exclusive Husband of 911 victim goes to Gitmo to spare plotters from death sentence NY Post The husband of a woman killed on 911 went to Guantanamo Bay on a shocking secret mission — to try to save the lives of the alQaeda monsters who planned the murder Blake Allison — one of 10 relatives of victims to win a lottery for tickets to the arraignment of confessed 911 mastermind Khalid Sheik Mohammed and four of his evil accomplices — had told people he was making the trip because “I wanted to see the faces of the people accused of murdering my wife’’ But while there the 62yearold winecompany executive held a clandestine meeting with the terrorists’ lawyers in which he offered to testify against putting their clients to death AP NIGHTMARE Anna Allison was aboard one of the jets flown into the Twin Towers in a plot orchestrated by Khalid Sheik Mohammed above Now her husband wants to help him NIGHTMARE Anna Allison was aboard one of the jets flown into the Twin Towers in a plot orchestrated by Khalid Sheik Mohammed Now her husband wants to help him A vocal critic of capital punishment Allison wants to convince the US government to spare the lives of KSM and his minions even if a military commission convicts them of a slew of deathpenalty charges “The public needs to know there are family members out there who do not hold the view that these men should be put to death” Allison told The Post “We can’t kill our way to a peaceful tomorrow” Allison’s 48yearold wife Anna was a software consultant on her way to visit a client in Los Angeles when her plane American Airlines Flight 11 was smashed into World Trade Center Tower 1 on Sept 11 2001 In a lengthy conversation from his home in New Hampshire Allison explained his controversial view — one he admits is not shared by his late wife’s relatives or by the other family members of victims he met at Guantanamo “My opposition to the death penalty does not say I don’t want the people who killed my wife and the other 911 victims brought to account for their crimes” he said “But for me opposition to the death penalty is not situational Just because I was hurt very badly and personally does not in my mind give me the goahead to take a life” He said that “911 was a particularly egregious and appalling crime” but added “I just think it’s wrong to take a life” Allison who has remarried is under no illusion that the terrorists have reformed — and would not gladly kill more Americans After staring at the fiendish faces of KSM Ramzi bin al Shibh Walid bin Attash Mustafa alHawsawi and KSM nephew Ali Abdul Aziz Ali Allison said he is certain they have “no apparent remorse and would do it again” Still he said “I’ve been opposed to the death penalty for decades before my wife was murdered on 911 “I’m still opposed to it” He said he spoke to other family members at Guantanamo and came to realize he was alone in his view" 
    }

###### Response

    {
        "_id": "1",
        "confidence": "0.8731268",
        "predictions": "reliable"
    }


###### API Latency Analysis

    Elasped time for predicting 100 articles is 64.732269 secs

    


### TODO List

  - Apply version control for every instance of concept drift metrics
  - Integrate data pipeline
  - Add documentation for DVC pull of data version and model version in
    continuous learning

#### Continuous Learner

###### Install Apex package for Mixed-Precision Deep Learning

    git clone https://github.com/NVIDIA/apex.git
    cd apex
    pip install -v --no-cache-dir ./
    cd ..
