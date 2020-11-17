import pandas as pd



cats = {}


def smooth_categories(dataframe, categories, multi_class):
    """
    This function takes a dataset and sets the number of samples of each
    class to the specified in the categories param
    :param dataframe (pandas DF)
    :param categories (dict) dictionary with the number of samples desired per class
    :multi_class (DF col) column of multiclass labels
    """
    #Copy dataframe structure
    df = dataframe.iloc[0:0]
    
    for cat in categories:
        selection = dataframe[dataframe[multi_class] == cat]
        count = selection[multi_class].count()
        if count >= categories[cat]:
            df = pd.concat([df, selection.head(categories[cat])])
        else:
            #Add n times the block
            block = dataframe.iloc[0:0]
            times = int(categories[cat] / count)    
            for i in range(times+1):
                block = pd.concat([block, selection])
            #Remove the remaining
            df = pd.concat([df, block.head(categories[cat])])
    
    return df