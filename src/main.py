from extract_face_feature import get_texture,df_clean,model_training
from argparse import ArgumentParser
import gc
from tqdm import tqdm , tqdm_pandas
import pandas as pd 

def main(args):
    ##### 把整理好的資料進行特徵萃取 #######
    df_index = pd.read_pickle(args.file_path)
    tqdm.pandas()
    block = df_index['index'].progress_apply(lambda x : get_texture(df_index.iloc[x],image_size))
    df_index = pd.concat([df_index,block],axis=1)
    df_index.to_pickle(args.storage_dir)
    del df_index
    gc.collect()
    
    main_reference = pd.read_pickle(args.storage_dir)
    model_training(df_clean(main_reference))




if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_path",
                        default='block/df_index_balanced_20221028.pkl',
                        help="The location of data which store data index and correspond information.",
                        type=str)
    parser.add_argument("--storage_dir",
                        default='block/df_index_balanced_main_reference_reproduce.pkl',
                        help="The location to store data.",
                        type=str)
    args = parser.parse_args()
    main(args)