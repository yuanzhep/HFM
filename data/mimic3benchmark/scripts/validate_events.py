from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
from tqdm import tqdm

def is_subject_folder(x):
    return str.isdigit(x)

def main():
    n_events = 0                   
    empty_hadm = 0                 
    no_hadm_in_stay = 0            
    no_icustay = 0                 
    recovered = 0                  
    could_not_recover = 0         
    icustay_missing_in_stays = 0   

    parser = argparse.ArgumentParser()
    parser.add_argument('subjects_root_path', type=str,
                        help='Directory containing subject subdirectories.')
    args = parser.parse_args()
    print(args)

    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))

    for subject in tqdm(subjects, desc='Iterating over subjects'):
        stays_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'stays.csv'), index_col=False,
                               dtype={'HADM_ID': str, "ICUSTAY_ID": str})
        stays_df.columns = stays_df.columns.str.upper()

        assert(not stays_df['ICUSTAY_ID'].isnull().any())
        assert(not stays_df['HADM_ID'].isnull().any())
        assert(len(stays_df['ICUSTAY_ID'].unique()) == len(stays_df['ICUSTAY_ID']))
        assert(len(stays_df['HADM_ID'].unique()) == len(stays_df['HADM_ID']))

        events_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index_col=False,
                                dtype={'HADM_ID': str, "ICUSTAY_ID": str})
        events_df.columns = events_df.columns.str.upper()
        n_events += events_df.shape[0]

        empty_hadm += events_df['HADM_ID'].isnull().sum()
        events_df = events_df.dropna(subset=['HADM_ID'])

        merged_df = events_df.merge(stays_df, left_on=['HADM_ID'], right_on=['HADM_ID'],
                                    how='left', suffixes=['', '_r'], indicator=True)

        no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
        merged_df = merged_df[merged_df['_merge'] == 'both']

        cur_no_icustay = merged_df['ICUSTAY_ID'].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, 'ICUSTAY_ID'] = merged_df['ICUSTAY_ID'].fillna(merged_df['ICUSTAY_ID_r'])
        recovered += cur_no_icustay - merged_df['ICUSTAY_ID'].isnull().sum()
        could_not_recover += merged_df['ICUSTAY_ID'].isnull().sum()
        merged_df = merged_df.dropna(subset=['ICUSTAY_ID'])

        icustay_missing_in_stays += (merged_df['ICUSTAY_ID'] != merged_df['ICUSTAY_ID_r']).sum()
        merged_df = merged_df[(merged_df['ICUSTAY_ID'] == merged_df['ICUSTAY_ID_r'])]

        to_write = merged_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']]
        to_write.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)

    assert(could_not_recover == 0)
    print('n_events: {}'.format(n_events))
    print('empty_hadm: {}'.format(empty_hadm))
    print('no_hadm_in_stay: {}'.format(no_hadm_in_stay))
    print('no_icustay: {}'.format(no_icustay))
    print('recovered: {}'.format(recovered))
    print('could_not_recover: {}'.format(could_not_recover))
    print('icustay_missing_in_stays: {}'.format(icustay_missing_in_stays))

if __name__ == "__main__":
    main()