import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import folium
from folium import plugins

model = pickle.load(open('pickles/randomforest_10.pkl','rb'))
ssX = pickle.load(open('pickles/ssX.pkl','rb'))
df = pickle.load(open('pickles/df_summary.pkl','rb'))
col_order = pickle.load(open('pickles/X_col_log.pkl','rb'))[1:]

def gen_data(month,msqto_qnt,spray,erraticus,pipiens,pipiens_restuans,restuans,salinarius,tarsalis,territans):
    df_input = df[df['month']==month].reset_index(drop=True).copy()

    # hardcoded for use in web app as options available for user input
    if msqto_qnt == 0.5:
        df_input['msqto_cnt'] = df_input['msqto_avg']
    elif msqto_qnt == 0.1:
        df_input['msqto_cnt'] = df_input['msqto_low']
    else:
        df_input['msqto_cnt'] = df_input['msqto_high']

    df_input['msqto_log'] = np.log(0.001+df_input['msqto_cnt'])
    df_input['spray_targeted'] = np.ones((df_input.shape[0],1)) if spray == 1 else np.zeros((df_input.shape[0],1))
    df_input['spray_targeted_prev'] = np.ones((df_input.shape[0],1)) if spray == 1 else np.zeros((df_input.shape[0],1))
    df_new = pd.DataFrame(columns=df_input.columns)

    species_dict = {'pipiens':pipiens,'pipiens_restuans':pipiens_restuans,'restuans':restuans,'salinarius':salinarius,'tarsalis':tarsalis,'territans':territans}

    for idx,val in enumerate(species_dict):
        df_loop = df_input.copy()
        np_species = np.zeros((df_loop.shape[0],len(species_dict)))
        if species_dict[val]==1:
            np_species[:,idx] = species_dict[val]
            df_species = pd.DataFrame(np_species,columns=['species_culex_'+x for x in species_dict.keys()])
            df_loop = df_loop.join(df_species)
            df_new = pd.concat((df_new,df_loop),axis=0)

    if erraticus == 1:
        np_species = np.zeros((df_loop.shape[0],len(species_dict)))
        df_loop = df_input.copy()
        df_species = pd.DataFrame(np_species,columns=['species_culex_'+x for x in species_dict.keys()])
        df_loop = df_loop.join(df_species)
        df_new = pd.concat((df_new,df_loop),axis=0)

    df_new.drop_duplicates().reset_index(drop=True,inplace=True)

    X_scaled = ssX.transform(df_new[col_order])

    # prob of 1
    y_pred = model.predict_proba(X_scaled)[:,1]

    df_new['virus_present'] = y_pred

    df_output = df_new[['month','latitude','longitude','virus_present']].copy()
    df_output = df_output.groupby(['month','latitude','longitude'])['virus_present'].apply(np.mean).reset_index()
    df_output['percentile'] = df_output['virus_present'].transform(lambda x: x.rank(pct=True))
    return(df_output)

def gen_map(df_output, colors):

    #colors = ['#B7A2DB','#9B5A8A','#55286F']
    #colors = ['#9EFFB6','#47B2A9','#417271']

    chicago_map = folium.Map(location = [df_output['latitude'].mean(), df_output['longitude'].mean()], 
                             zoom_start = 10.5, tiles='Stamen Toner')

    for _,row in df_output.iterrows():
        color = colors[min(len(colors)-1,int(row['percentile']*len(colors)))]

        folium.CircleMarker(
                location = [row["latitude"], row["longitude"]],  # Note that NS comes before EW for folium!
                radius   = 20,
                fill = True,
                fill_color = color,
                color = color,
                fill_opacity = .9,
                opacity = 1
            ).add_to(chicago_map)

    return chicago_map

def identify_bins(df_output,colors):
    df_output['bin'] = df_output['percentile'].transform(lambda x:min(len(colors)-1,int(x*len(colors))))
    bins = (df_output[['virus_present','bin']].groupby(['bin'])
                                    .max().reset_index()
                                    .sort_values(by=['bin'],ascending=True)
                                    .values[:,1]
    )
    return bins