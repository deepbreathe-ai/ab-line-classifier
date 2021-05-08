
import yaml
import os
import pandas as pd
from tqdm import tqdm
import wget

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def data_pull(update):
    '''
    Pull raw clips from AWS database using a generated query in csv format
    :update:    a boolean value, used to determine if the current dataset should be updated, or completely downloaded
                affects the CSV_PATH variable
    '''

    CSV_PATH=''

    if update == '0':
        CSV_PATH = 'CLIPS_TABLE'
    elif update == '1':
        CSV_PATH = 'UPDATED_CLIPS_TABLE'
    else:
        print("Unknown command, exiting program.")

    output_folder = cfg['PATHS']['RAW_CLIPS']
    df = pd.read_csv(cfg['PATHS'][CSV_PATH])
    print('Getting AWS links...')
    # Dataframe of all clip links
    links = df.s3_path
    print('Fetching clips from AWS...')
    # Download clips and save to disk
    for link in tqdm(links):
        print(link)
        firstpos = link.rfind("/")
        lastpos = len(link)
        filename = link[firstpos+1:lastpos]
        wget.download(link, output_folder + filename)
    print('Fetched clips successfully!')
    return

if __name__ == '__main__':
    data_pull(1)