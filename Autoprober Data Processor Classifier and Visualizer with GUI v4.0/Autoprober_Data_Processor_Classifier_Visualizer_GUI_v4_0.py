#!/usr/bin/env python
# coding: utf-8

# # Autoprober Data Processor, Sensor Classifier & Visualizer GUI - Version 4.0

# If you have any doubt about the code fill free to contact-me at:<br>
# - daniel.h.c.sousa@tecnico.ulisboa.pt
# - danielsoussa@gmail.com

# ## Imports

# In[ ]:


import numpy as np
import pandas as pd
from math import ceil
from tkinter import *
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
from matplotlib.ticker import EngFormatter
import os, shutil, glob, re, ast, imp, datetime
from tkinter import ttk, filedialog, simpledialog, messagebox


# ## Processor

# ### Pre Processing data

# In[ ]:


def check_directories(self):
    parent_directory = self.save_directory
    smps = self.filesnames
    if parent_directory == '':
        parent_directory = None
    if smps == '':
        smps = None
    if parent_directory != None and smps == None:
        messagebox.showerror(title='ERROR',
                             message='No SMP files were selected,'+\
                             ' computations interrupted!')
        stop = 1
    elif parent_directory == None and smps != None:
        messagebox.showerror(title='ERROR',
                             message='No Saving Directory was selected,'+\
                             ' computations interrupted!')
        stop = 1
    elif parent_directory == None and smps == None:
        messagebox.showerror(title='ERROR',
                             message='No SMP files and Saving Directory were selected,'+\
                             ' computations interrupted!')
        stop = 1
    elif parent_directory != None and smps != None:
        stop = 0
    if len(os.listdir(parent_directory)) != 0 and stop == 0:
        empty_folder_quest = messagebox.askquestion(title = 'WARNING',
                                                    message = 'Saving Directory selected'+\
                                                    ' is not empty, computations MAY BE '+\
                                                    'interrupted if overwriting occours!'+\
                                                    ' Do you wish to continue?',
                                                    icon = 'warning')
        if empty_folder_quest != 'yes':
            stop = 1
    return stop

def check_directories_predictions(self):
    csvs_directory = self.csvs_directory
    saving_directory = self.save_directory_predictions
    model_directory = self.model
    if csvs_directory == '':
        csvs_directory = None
    if saving_directory == '':
        saving_directory = None
    if model_directory == '':
        model_directory = None
        
    if saving_directory != None and model_directory != None and csvs_directory == None:
        messagebox.showerror(title='ERROR',
                             message='No CSVs directory was selected, computations'+\
                             ' interrupted!')
        stop = 1
    elif saving_directory == None and model_directory != None and csvs_directory != None:
        messagebox.showerror(title='ERROR',
                             message='No Saving Directory was selected, computations'+\
                             ' interrupted!')
        stop = 1
    elif saving_directory != None and model_directory == None and csvs_directory != None:
        messagebox.showerror(title='ERROR',
                             message='No Model was selected, computations interrupted!')
        stop = 1     
    elif saving_directory == None and model_directory == None and csvs_directory != None:
        messagebox.showerror(title='ERROR',
                             message='No Saving Directory and Model were selected, '+\
                             'computations interrupted!')
        stop = 1
    elif saving_directory != None and model_directory == None and csvs_directory == None:
        messagebox.showerror(title='ERROR',
                             message='No Model and CSVs directory were selected, '+\
                             'computations interrupted!')
        stop = 1
    elif saving_directory == None and model_directory != None and csvs_directory == None:
        messagebox.showerror(title='ERROR',
                             message='No Saving Directory and CSVs directory were '+\
                             'selected, computations interrupted!')
        stop = 1    
    elif saving_directory == None and model_directory == None and csvs_directory == None:
        messagebox.showerror(title='ERROR',
                             message='No CSVs directory, Saving Directory and Model were '+\
                             'selected, computations interrupted!')
        stop = 1
    elif saving_directory != None and model_directory != None and csvs_directory != None:
        stop = 0
    if len(os.listdir(saving_directory)) != 0 and stop == 0:
        empty_folder_quest = messagebox.askquestion(title = 'WARNING',
                                                    message = 'Saving Directory selected is '+\
                                                    'not empty, computations MAY BE interrupted'+\
                                                    ' if overwriting occours! Do you wish to continue?',
                                                    icon = 'warning')
        if empty_folder_quest != 'yes':
            stop = 1
            
    try:
        imp.find_module('keras')
        stop = 0
    except:
        messagebox.showerror(title='ERROR',
                             message='Keras is not installed, computations interrupted!')
        stop = 1
    return stop

def process_data(file_name, calib_H, map_index_name = ' Input 3 Value',
                 field_current_name = ' Input 1 Value',
                 probes_current_name = ' Input 2 Value',
                 probes_voltage_name = ' M4 Volt [V]'):
    data_smp = pd.read_csv(file_name, sep = ';')
    df = data_smp[[map_index_name, field_current_name, probes_current_name, probes_voltage_name]]
    df.columns = ['Map Index', 'Field Current [A]', 'Probes Current [A]', 'Probes Voltage [V]']
    df.insert(4, 'Resistance [Ω]', df['Probes Voltage [V]']/df['Probes Current [A]'])
    df.insert(5, 'Applied Field [Oe]', df['Field Current [A]']*calib_H)
    df.insert(6, 'Bias Current [A]', df['Probes Current [A]'])
    num_sensors = df['Map Index'].max()
    return df, num_sensors


# ### Functions - Processor

# #### R(H)

# In[ ]:


def make_RH(num_sensors, df, saving_name, saving_directory, self):
    self.progress_text_inter.set('->Working on R(H).')

    os.makedirs(saving_directory + '/R_H_curves/SMP/R_H_' + saving_name)
    os.makedirs(saving_directory + '/R_H_curves/SMP/R_H_' + saving_name + '/Data')
    self.progress_bar_inter['maximum'] = num_sensors+1
    for index in range(num_sensors+1):
        self.progress_bar_inter['value'] = index
        self.progress_bar_inter.update()
        df_prov = df.loc[df['Map Index'] == index]

        df_prov.to_csv(saving_directory + '/R_H_curves/SMP/R_H_' + saving_name +                       '/Data/map_index_'+ str(index) +'.csv',
                       columns = ['Field Current [A]', 'Applied Field [Oe]',
                                  'Resistance [Ω]', 'Bias Current [A]'], index = False)
        df_prov.plot(kind='scatter', x='Applied Field [Oe]', y='Resistance [Ω]', color='red')
        plt.grid()
        ax = plt.gca()
        plt.ylabel('Resistance '+ r'[$\Omega$]')
        formatter = EngFormatter(places=1, sep='\N{THIN SPACE}')
        ax.yaxis.set_major_formatter(formatter)
        ibias = df_prov['Bias Current [A]']
        ibias = ibias.to_numpy()
        if (ibias[0] == ibias).all():
            plt.title('Map Index = ' + str(index) + '\n' + r'$I_{bias} = $' +                      np.format_float_scientific(ibias[0], precision = 3,
                                                 trim = '0', exp_digits = 1) + ' [A]')
        else:
            plt.title('Map Index = ' + str(index))
        plt.tick_params(direction='in', top=True, right=True)
        plt.tight_layout()
        plt.savefig(saving_directory + '/R_H_curves/SMP/R_H_' + saving_name +                    '/map_index_'+ str(index) +'.png')
        plt.close()
    self.progress_bar_inter['value'] = 0
    self.progress_bar_inter.update()
    return None


# #### MR

# In[ ]:


def make_MR(num_sensors, df, saving_name, saving_directory, self, write_xslx = 1,
            threshold_R_sc = 100, threshold_R_oc = 10**8, threshold_MR = 400):
    self.progress_text_inter.set('->Working on MR.')

    n_to_average = 3
    if not os.path.exists(saving_directory + '/MR'):
        os.makedirs(saving_directory + '/MR')
    df_MR = pd.DataFrame(columns = ['Map Index', 'MR [%]'])
    self.progress_bar_inter['maximum'] = num_sensors+1
    for index in range(num_sensors+1):
        self.progress_bar_inter['value'] = index
        self.progress_bar_inter.update()
        df_prov = df.loc[df['Map Index'] == index]
        if df_prov['Resistance [Ω]'].nsmallest(n_to_average).mean() == 0:
            min_R = 0.00000000000001
        else:
            min_R = df_prov['Resistance [Ω]'].nsmallest(n_to_average).mean()
        if df_prov['Resistance [Ω]'].nlargest(n_to_average).mean() == 0:
            max_R = 0.00000000000001
        else:
            max_R= df_prov['Resistance [Ω]'].nlargest(n_to_average).mean()
            
        MR = (max_R-min_R)*100.0/(min_R)

        #Remove too much high MR like > 400
        if MR > threshold_MR:
            MR = np.nan
            
         #MR is classified as SC only if the average R is lower then the threshold
        if df_prov['Resistance [Ω]'].mean() < threshold_R_sc:
            MR = 'SC'
        #MR is classified as OC only if the average R is bigger then the threshold
        if df_prov['Resistance [Ω]'].mean() > threshold_R_oc:
            MR = 'OC'
        
        df1 = pd.DataFrame([[index, MR]], columns = ['Map Index', 'MR [%]'])
        df_MR = df_MR.append(df1, ignore_index=True)
    if write_xslx == 1:
        df_MR.to_excel(saving_directory + '/MR/MR_' + saving_name + '.xlsx', index = False)
    self.progress_bar_inter['value'] = 0
    self.progress_bar_inter.update()
    return df_MR


# #### R at certain H

# In[ ]:


def make_R_at_H(num_sensors, df, saving_name, saving_directory, self, H = 0,
                write_xslx = 0, threshold_R_sc = 100, threshold_R_oc = 10**8):
    #H=0 by predifinition
    self.progress_text_inter.set('->Working on R at field = ' + str(H))
    
    if not os.path.exists(saving_directory + '/R_at_H' + str(H)):
        os.makedirs(saving_directory + '/R_at_H' + str(H))
    
    df_R_at_H = pd.DataFrame(columns = ['Map Index', 'Applied Field [Oe]',
                                        'Resistance [Ω]'])
    df_R_at_H_avg = pd.DataFrame(columns = ['Map Index', 'Applied Field [Oe]',
                                            'Resistance [Ω]'])
    self.progress_bar_inter['maximum'] = num_sensors+1
    for index in range(num_sensors+1):
        self.progress_bar_inter['value'] = index
        self.progress_bar_inter.update()
        df_prov = df.loc[df['Map Index'] == index]
        df_sort = df_prov.iloc[(df_prov['Applied Field [Oe]'] - H).abs().argsort()[:2]]
        df_A = df_sort[['Map Index', 'Applied Field [Oe]', 'Resistance [Ω]']]
        
        mean_R = df_A['Resistance [Ω]'].mean()

        df_R_at_H = df_R_at_H.append(df_A, ignore_index=True)
        
        if df_prov['Resistance [Ω]'].mean() < threshold_R_sc:
            mean_R = 'SC'
        if df_prov['Resistance [Ω]'].mean() > threshold_R_oc:
            mean_R = 'OC'

        new_row = [{'Map Index':int(df_A['Map Index'].mean()),
                    'Applied Field [Oe]':df_A['Applied Field [Oe]'].mean(),
                    'Resistance [Ω]':mean_R}]
        df_R_at_H_avg = df_R_at_H_avg.append(new_row, ignore_index = True, sort = False)
    df_R_at_H_avg.to_excel(saving_directory + '/R_at_H' + str(H) + '/' + saving_name +                           '_avgs.xlsx', index = False)
    if write_xslx == 1:
        df_R_at_H.to_excel(saving_directory + '/R_at_H' + str(H) + '/' + saving_name +                           '_both.xlsx', index = False)
    self.progress_bar_inter['value'] = 0
    self.progress_bar_inter.update()
    return df_R_at_H, df_R_at_H_avg


# #### Rmin

# In[ ]:


def make_Rmin(num_sensors, df, saving_name, saving_directory, self,
              threshold_R_sc = 100, threshold_R_oc = 10**8):
    self.progress_text_inter.set('->Working on Rmin.')
    n_to_average = 3
    if not os.path.exists(saving_directory + '/Rmin'):
        os.makedirs(saving_directory + '/Rmin')
    df_Rmin = pd.DataFrame(columns = ['Map Index', 'Rmin [Ω]'])
    self.progress_bar_inter['maximum'] = num_sensors+1
    for index in range(num_sensors+1):
        self.progress_bar_inter['value'] = index
        self.progress_bar_inter.update()
        df_prov = df.loc[df['Map Index'] == index]
        Rmin = df_prov['Resistance [Ω]'].nsmallest(n_to_average).mean()
        threshold_R_sc = float(threshold_R_sc)
        threshold_R_oc = float(threshold_R_oc)

        if df_prov['Resistance [Ω]'].mean() < threshold_R_sc:
            Rmin = 'SC'
        if df_prov['Resistance [Ω]'].mean() > threshold_R_oc:
            Rmin = 'OC'
        
        df1 = pd.DataFrame([[index, Rmin]], columns = ['Map Index', 'Rmin [Ω]'])
        df_Rmin = df_Rmin.append(df1, ignore_index=True)
    df_Rmin.to_excel(saving_directory + '/Rmin/Rmin_' + saving_name + '.xlsx',
                     index = False)
    self.progress_bar_inter['value'] = 0
    self.progress_bar_inter.update()
    return df_Rmin


# #### Make groups by index

# In[ ]:


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def group_by_index(self):
    self.progress_text_inter.set('->Grouping figures, progress bar not available.')
    stop = 0
    owd = os.getcwd()
    parent_directory = self.save_directory
    subdirectories_list = [x[0] for x in os.walk(parent_directory + '\\R_H_curves\\SMP')][1:]

    all_folders_info = []
    all_folders_names_info = []
    for subdirectorie in subdirectories_list:
        os.chdir(subdirectorie)
        index_in_folder = []
        names_fig_in_folder = glob.glob('*.png')
        names_fig_in_folder.sort(key=natural_keys)
        all_folders_names_info.append(names_fig_in_folder)
        for image in glob.glob('*.png'):
            res = ''.join(filter(lambda i: i.isdigit(), image))
            index_in_folder.append(int(res))
            os.chdir(owd)
        index_in_folder.sort()
        all_folders_info.append(index_in_folder)
    all_folders_info = [x for x in all_folders_info if x != []]
    it = iter(all_folders_info)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        stop = 1
        messagebox.showwarning(title = 'WARNING',
                               message = 'Not all indexes of SMP files have the same maximum' +\
                               ' value, it is impossible to group by index!')
    if stop == 0:
        for i in all_folders_info[0]:
            os.makedirs(parent_directory + '/R_H_curves/Index/Index_' + str(i))

        for i in range(len(subdirectories_list)):
            for j in range(len(all_folders_names_info[i])):
                name = subdirectories_list[i].replace(subdirectories_list[i].rsplit('\\', 1)[0], '')
                shutil.copy(subdirectories_list[i] + '\\' + all_folders_names_info[i][j],
                            parent_directory + '\\R_H_curves\\Index\\Index_' + str(j) + name +'.png')


# ## Classifier

# ### Functions

# In[ ]:


def find_csv_filenames(path_to_dir, suffix = '.csv' ):
    filenames = os.listdir(path_to_dir)
    filenames.sort(key = natural_keys)
    return [filename for filename in filenames if filename.endswith(suffix)]

def import_csvs_data(path_to_dir, filenames, self):
    self.progress_text_classi.set('->Importing CSVs.')
    self.progress_bar_classi['maximum'] = len(filenames) + 1
    list_dfs = []
    for index in range(len(filenames)):
        self.progress_bar_classi['value'] = index
        self.progress_bar_classi.update()
        filename = filenames[index]
        df = pd.read_csv(path_to_dir + '/' + filename,
                     usecols = ['Applied Field [Oe]',
                                'Resistance [Ω]'])
        list_dfs.append(df)
    self.progress_bar_classi['value'] = 0
    self.progress_bar_classi.update()
    self.progress_text_classi.set('')
    return list_dfs

def resize(arr, lower = -1.0, upper = 1.0):
    arr_i = arr.copy()
    for n in range(len(arr)):
        arr[n] = (upper - lower)/(max(arr_i) - min(arr_i)) * (arr[n] - max(arr_i)) + upper
    return arr

def linear_inter(list_x, list_y):
    list_final_x = []
    list_final_y = []
    for i in range(len(list_x)-1):
        list_final_x.append(list_x[i])
        interpol_x_value = (list_x[i] + list_x[i+1])/2.0
        list_final_x.append(interpol_x_value)
        list_final_y.append(list_y[i])
        interpol_y_value = (list_y[i] + list_y[i+1])/2.0
        list_final_y.append(interpol_y_value)
    list_final_x.append(list_x[-1])
    list_final_y.append(list_y[-1])
    return list_final_x, list_final_y

def upsampling(df_x, df_y, goal_n = 70):
    i = 0
    list_in_x = df_x.tolist()
    list_in_y = df_y.tolist()
    while 2*len(list_in_x) <= goal_n :
        i+=1
        list_x, list_y = linear_inter(list_in_x, list_in_y)
        list_in_x = list_x
        list_in_y = list_y
    
    #Deal with times where there are no need to interpolate all points
    if 2*len(list_in_x) > goal_n:
        n_to_inter = goal_n - len(list_in_x) #number of points to add
        mult = int(len(list_in_x)/n_to_inter) #at each multiple of this add a inter
        list_new_x = []
        list_new_y = []
        k=0
        for i in range(len(list_in_x)):
            list_new_x.append(list_in_x[i])
            list_new_y.append(list_in_y[i])
            if i % mult == 0 and i != 0 and k < n_to_inter and i+1 < len(list_in_x): #if multiple add inter
                #mult #in the first dont # while havent achieved n_to inter
                #and do not try to interpolate with value that does not exist
                list_new_x.append((list_in_x[i] + list_in_x[i+1])/2.0) #inter value
                list_new_y.append((list_in_y[i] + list_in_y[i+1])/2.0) #inter value
                k += 1
        list_x = list_new_x
        list_y = list_new_y
    
    if goal_n - len(list_x) == 1:
        list_x.insert(1, (list_in_x[0] + list_in_x[1])/2.0)
        list_y.insert(1, (list_in_y[0] + list_in_y[1])/2.0)

    s_out_x = pd.Series(list_x)
    s_out_y = pd.Series(list_y)
    return s_out_x, s_out_y

def downsampling(df_x, df_y, goal_n = 70):
    list_in_x = df_x.tolist()
    list_in_y = df_y.tolist()
    length = float(len(list_in_x))
    list_out_x = []
    list_out_y = []
    for i in range(goal_n):
        list_out_x.append(list_in_x[int(ceil(i * length / goal_n))])
        list_out_y.append(list_in_y[int(ceil(i * length / goal_n))])
    s_out_x = pd.Series(list_out_x)
    s_out_y = pd.Series(list_out_y)
    return s_out_x, s_out_y

def treat_csv_data(list_dfs, self, goal_n_points = 70):
    list_dfs_treated = []
    list_data_final = []
    self.progress_text_classi.set('->Treating CSVs.')
    self.progress_bar_classi['maximum'] = len(list_dfs) + 1
    for index in range(len(list_dfs)):
        self.progress_bar_classi['value'] = index
        self.progress_bar_classi.update()
        df = list_dfs[index]
        if df.shape[0] < goal_n_points:
            s_out_app, s_out_resis =  upsampling(df['Applied Field [Oe]'],
                                                 df['Resistance [Ω]'],
                                                 goal_n = goal_n_points)
            df = pd.DataFrame({'Applied Field [Oe]' : s_out_app, 'Resistance [Ω]' : s_out_resis})
            df = df.dropna()
            
        if df.shape[0] > goal_n_points:
            df['Applied Field [Oe]'], df['Resistance [Ω]'] = downsampling(df['Applied Field [Oe]'],
                                                                          df['Resistance [Ω]'],
                                                                          goal_n = goal_n_points)
            df = df.dropna()
        
        df['Resized H [Oe]'] = resize(df['Applied Field [Oe]'])
        df['Resized R [Ω]'] = resize(df['Resistance [Ω]'])
        list_dfs_treated.append(df.drop(['Applied Field [Oe]', 'Resistance [Ω]'], axis = 1))
    for df in list_dfs_treated:
        df_final_to_list = df.values.tolist()
        list_data_final.append(df_final_to_list)
    data_final_array = np.asarray(list_data_final, dtype = np.float32)
    data_final_array = data_final_array.reshape(-1, goal_n_points, 2, 1)
    data_final_array = data_final_array.astype('float32')
    self.progress_bar_classi['value'] = 0
    self.progress_bar_classi.update()
    self.progress_text_classi.set('')
    return data_final_array

def predict_proba_to_oneh(predict_prob, self):
    encoded_array = np.empty((0, 4), int)
    self.progress_text_classi.set('->Making predictions (encoded).')
    self.progress_bar_classi['maximum'] = len(predict_prob) + 1
    for index in range(len(predict_prob)):
        self.progress_bar_classi['value'] = index
        self.progress_bar_classi.update()
        i = predict_prob[index]
        if i[0] > i[1] and i[0] > i[2] and i[0] > i[3]:
            l = np.array([[1, 0, 0, 0]])
        elif i[1] > i[0] and i[1] > i[2] and i[1] > i[3]:
            l = np.array([[0, 1, 0, 0]])
        elif i[2] > i[0] and i[2] > i[1] and i[2] > i[3]:
            l = np.array([[0, 0, 1, 0]])
        elif i[3] > i[0] and i[3] > i[1] and i[3] > i[2]:
            l = np.array([[0, 0, 0, 1]])
        else:
            print('Function must be wrong.')
        encoded_array = np.append(encoded_array, l, axis=0)
    self.progress_bar_classi['value'] = 0
    self.progress_bar_classi.update()
    self.progress_text_classi.set('')
    return encoded_array

def decoder(y_array_encoded, self):
    decoded_list = []
    self.progress_text_classi.set('->Decoding one-hot predictions.')
    self.progress_bar_classi['maximum'] = len(y_array_encoded) + 1
    for i in range(len(y_array_encoded)):
        self.progress_bar_classi['value'] = i
        self.progress_bar_classi.update()
        if np.array_equal(y_array_encoded[i], np.array([1, 0, 0, 0])):
            l = 'OK'
        elif np.array_equal(y_array_encoded[i], np.array([0, 1, 0, 0])):
            l = 'M'
        elif np.array_equal(y_array_encoded[i], np.array([0, 0, 1, 0])):
            l = 'A'
        elif np.array_equal(y_array_encoded[i], np.array([0, 0, 0, 1])):
            l = 'NOK'
        decoded_list.append(l)
    decoded_array = np.array(decoded_list)
    self.progress_bar_classi['value'] = 0
    self.progress_bar_classi.update()
    self.progress_text_classi.set('')
    return decoded_array
 
def make_predictions_from_path(path_to_csvs, path_to_model, saving_directory,
                               predi_excel_name, self):
    #saving_directory = path_to_csvs #save the prediction inside the csvs folder
    from keras.models import load_model
    if not os.path.exists(saving_directory + '/Predictions'): #if folder does not exist
        os.makedirs(saving_directory + '/Predictions') #creat it
    model = load_model(path_to_model) #loading model
    filenames = find_csv_filenames(path_to_csvs) #get all filenames inside directory
    df_csvs = import_csvs_data(path_to_csvs, filenames, self) #import all csvs to list of dataframes
    data_array = treat_csv_data(df_csvs, self) #up/downsampling and shaping to input layer
    y_pred_oneh = predict_proba_to_oneh(model.predict(data_array), self) #making prediction
    y_pred_decoded = decoder(y_pred_oneh, self) #decoding prediction from 1h to labels
    self.progress_text_classi.set('->Saving predictions.')
    df_y_pred = pd.DataFrame({'Map Index': list(range(len(y_pred_decoded))),
                              'Prediction': y_pred_decoded,
                              'OK' : model.predict(data_array)[:, 0],
                              'M' : model.predict(data_array)[:, 1],
                              'A' : model.predict(data_array)[:, 2],
                              'NOK' : model.predict(data_array)[:, 3]}) #making df
    df_y_pred.to_excel(saving_directory + '/Predictions/' + predi_excel_name + '.xlsx',
                       index = False) #saving df
    self.progress_text_classi.set('')


# ## Data vizualizer

# ### Functions

# In[ ]:


def get_all_name(path, extension):
    owd = os.getcwd()
    os.chdir(path)
    files_names = glob.glob('*' + extension)
    os.chdir(owd)
    full_path_list = []
    for i in files_names:
        full_path_list.append(path + '/' + i)
    return full_path_list

def get_rele_data(list_map_df, list_meas_df, relevant_init, relevant_final, quantity):
    list_rele_df = []
    for i in range(len(list_map_df)):
        map_df = list_map_df[i]
        meas_df = list_meas_df[i]
        map_df_relevant = map_df.iloc[relevant_init:relevant_final]
        meas_df_relevant = meas_df.iloc[relevant_init:relevant_final]
        relevant_df = pd.concat([map_df_relevant, meas_df_relevant['Map Index'],
                                 meas_df_relevant[quantity]], axis=1)
        list_rele_df.append(relevant_df)
    return list_rele_df

def get_rele_data_predi(list_map_df, list_meas_df, relevant_init, relevant_final):
    list_rele_df = []
    for i in range(len(list_map_df)):
        map_df = list_map_df[i]
        meas_df = list_meas_df[i]
        map_df_relevant = map_df.iloc[relevant_init:relevant_final]
        meas_df_relevant = meas_df.iloc[relevant_init:relevant_final]
        relevant_df = pd.concat([map_df_relevant, meas_df_relevant['Map Index'],
                                 meas_df_relevant['Prediction'], meas_df_relevant['OK'],
                                 meas_df_relevant['M'], meas_df_relevant['A'],
                                 meas_df_relevant['NOK']], axis=1)
        list_rele_df.append(relevant_df)
    return list_rele_df

def change_coordinates_put_name(sns_coors, dfs, names):
    '''
    Ensures: Returns the dataframe with absolute coordinates
    Requires: df = dataframe with measurements and relative coordinates
              sns_coor = absolute coordinates of sensor with respect to mask/map
    '''
    for i in range(len(names)):
        df = dfs[i]
        sns_coor = sns_coors[i]
        name = names[i]
        df['x abs'] = df['X'] + sns_coor[0]
        df['x abs'] = pd.to_numeric(df['x abs'])
        df['y abs'] = df['Y'] + sns_coor[1]
        df['y abs'] = pd.to_numeric(df['y abs'])
        df['quantity'] = name
        dfs[i] = df
    return dfs

def myround(x, base=10):
    #if isinstance(x, str):
        
    if 0 < x % 10 < 5:
        x += 5
    return base * round(x/base)


# ### Dies

# In[ ]:


def plot_die(n_dies_x, n_dies_y, x_o, y_o, chip_x_dim, chip_y_dim, fig):
    
    if n_dies_x < 0 and n_dies_y > 0:
        for i in range(0, n_dies_x, -1):
            for j in range(n_dies_y):
                fig.add_shape(type = 'rect', x0 = x_o + (i)*chip_x_dim, y0 = y_o + (j)*chip_y_dim,
                              x1 = x_o+ (i+1)*chip_x_dim, y1 = y_o + (j+1)*chip_y_dim,
                              line = dict(color = 'Black', width = 2))
    elif n_dies_x > 0 and n_dies_y < 0:
        for i in range(n_dies_x):
            for j in range(0, n_dies_y, -1):
                fig.add_shape(type = 'rect', x0 = x_o + (i)*chip_x_dim, y0 = y_o + (j)*chip_y_dim,
                              x1 = x_o+ (i+1)*chip_x_dim, y1 = y_o + (j+1)*chip_y_dim,
                              line = dict(color = 'Black', width = 2))
    elif n_dies_x < 0 and n_dies_y < 0:
        for i in range(0, n_dies_x, -1):
            for j in range(0, n_dies_y, -1):
                fig.add_shape(type = 'rect', x0 = x_o + (i)*chip_x_dim, y0 = y_o + (j)*chip_y_dim,
                              x1 = x_o+ (i+1)*chip_x_dim, y1 = y_o + (j+1)*chip_y_dim,
                              line = dict(color = 'Black', width = 2))
    else:
        for i in range(n_dies_x):
            for j in range(n_dies_y):
                fig.add_shape(type = 'rect', x0 = x_o + (i)*chip_x_dim, y0 = y_o + (j)*chip_y_dim,
                              x1 = x_o+ (i+1)*chip_x_dim, y1 = y_o + (j+1)*chip_y_dim,
                              line = dict(color = 'Black', width = 2))


# ## GUI

# In[ ]:


def close_window(): 
    root.destroy()
        

class App:
    def __init__(self, root):
        self.filesnames = None
        self.save_directory = None
        
        tabControl = ttk.Notebook(root)
        tab_1 = Frame(tabControl)
        tabControl.add(tab_1, text = 'Data Processor')

        
        ###DATA PROCESSOR
        frm_data_processor_0 = Frame(tab_1)
        
        ### Index Column
        self.lmap_ind_n = Label(frm_data_processor_0, text='Index Column:')
        self.lmap_ind_n.pack(side=TOP, fill='both')
        self.map_ind_n = Entry(frm_data_processor_0)
        self.map_ind_n.insert(END, ' Input 3 Value')
        self.map_ind_n.pack(side=TOP, fill='both')
        
        ### Field Current Column
        self.lfield_c_ind_n = Label(frm_data_processor_0, text='Field Current Column:')
        self.lfield_c_ind_n.pack(side=TOP, fill='both')
        self.field_c_ind_n = Entry(frm_data_processor_0)
        self.field_c_ind_n.insert(END, ' Input 1 Value')
        self.field_c_ind_n.pack(side=TOP, fill='both')
        
        ### Probes Current Column
        self.lprobes_c_ind_n = Label(frm_data_processor_0, text='Probes Current Column:')
        self.lprobes_c_ind_n.pack(side=TOP, fill='both')
        self.probes_c_ind_n = Entry(frm_data_processor_0)
        self.probes_c_ind_n.insert(END, ' Input 2 Value')
        self.probes_c_ind_n.pack(side=TOP, fill='both')
        
        ### Probes Voltage Column
        self.lprobes_v_ind_n = Label(frm_data_processor_0, text='Probes Voltage Column:')
        self.lprobes_v_ind_n.pack(side=TOP, fill='both')
        self.probes_v_ind_n = Entry(frm_data_processor_0)
        self.probes_v_ind_n.insert(END, ' M4 Volt [V]')
        self.probes_v_ind_n.pack(side=TOP, fill='both')
        
        
        #frm_data_processor_0.pack(side=LEFT)
        frm_data_processor_0.pack(expand = True, side = LEFT)
        frm_data_processor_1 = Frame(tab_1)
        
        ### Calibration Field
        self.lcalib_H = Label(frm_data_processor_1, text='Field calibration [Oe/A]:')
        self.lcalib_H.pack(side=TOP, fill='both')
        self.calib_H = Entry(frm_data_processor_1)
        self.calib_H.insert(END, 961.16)
        self.calib_H.pack(side=TOP, fill='both')

        ###Select SMP files
        self.smp = Button(frm_data_processor_1, text='Select SMPs', fg='black',
                          command=self.call_smp_names)
        self.smp.pack(side=TOP, fill='both')
        
        ###Select saving directory
        self.saving = Button(frm_data_processor_1, text='Select Saving Directory',
                             fg='black', command=self.call_save_directory)
        self.saving.pack(side=TOP, fill='both')
        
        ###Progress Bar SMP
        self.lprogress_bar_smp = Label(frm_data_processor_1, text='SMPs Progress:')
        self.lprogress_bar_smp.pack(side=TOP, fill='both')
        self.progress_bar_smp = ttk.Progressbar(frm_data_processor_1,
                                                orient = 'horizontal', length = 5,
                                                mode = 'determinate')
        self.progress_bar_smp.pack(side = TOP, fill = 'both')
        
        ###Progress Bar Intermediate
        self.lprogress_bar_inter = Label(frm_data_processor_1, text='Current Step:')
        self.lprogress_bar_inter.pack(side=TOP, fill='both')
        
        self.progress_text_inter = StringVar('') 
        self.progress_text_inter_field = Entry(frm_data_processor_1,
                                               textvariable = self.progress_text_inter)
        self.progress_text_inter_field.pack(side = TOP, fill = 'both') 
        
        self.progress_bar_inter = ttk.Progressbar(frm_data_processor_1,
                                                  orient = 'horizontal',
                                                  length = 5, mode = 'determinate')
        self.progress_bar_inter.pack(side = TOP, fill = 'both')

        #frm_data_processor_1.pack(side=LEFT)
        frm_data_processor_1.pack(expand = True, side = LEFT)
        frm_data_processor_2 = Frame(tab_1)
       
        ###BOOLEANS!
        
        ### Threshold open circuit
        self.lHigh_R = Label(frm_data_processor_2, text='OC [R] threshold [\u03a9]:')
        self.lHigh_R.pack(side=TOP, fill='both')
        self.High_R = Entry(frm_data_processor_2)
        self.High_R.insert(END, 1000000)
        self.High_R.pack(side=TOP, fill='both')

        ### Threshold short circuit
        self.lLow_R = Label(frm_data_processor_2, text='SC [R] threshold [\u03a9]:')
        self.lLow_R.pack(side=TOP, fill='both')
        self.Low_R = Entry(frm_data_processor_2)
        self.Low_R.insert(END, 100)
        self.Low_R.pack(side=TOP, fill='both')
        
        ###Title
        self.Lsecond_row = Label(frm_data_processor_2, text='Select functions:')
        self.Lsecond_row.pack(side=TOP, fill='both')
        
        #frm_RH = Frame(frm_data_processor_2)
        ###Make R(H)
        self.make_RH_bool = BooleanVar()
        self.make_RH_bool.set(True)
        self.Lmake_RH_bool = Checkbutton(frm_data_processor_2, text = 'Make R(H) plots',
                                         var = self.make_RH_bool) 
        self.Lmake_RH_bool.pack(side=TOP, anchor = NW)
        #self.Lmake_RH_bool.grid(row=1, column=3)
        #frm_RH.pack(side = LEFT)
        
        #frm_MR = Frame(frm_data_processor_2)
        ###Make MR
        self.make_MR_bool = BooleanVar() 
        self.make_MR_bool.set(True)
        self.Lmake_MR_bool = Checkbutton(frm_data_processor_2, text='Get MR',
                                         var = self.make_MR_bool) 
        self.Lmake_MR_bool.pack(side=TOP, anchor = NW)
        #frm_MR.pack(side = LEFT)

        ###Make Rmin
        self.make_Rmin_bool = BooleanVar() 
        self.make_Rmin_bool.set(True)
        self.Lmake_Rmin_bool = Checkbutton(frm_data_processor_2, text='Get R min',
                                           var = self.make_Rmin_bool) 
        self.Lmake_Rmin_bool.pack(side=TOP, anchor = NW)
        
        ###Make R AT H
        self.make_R_at_H_bool = BooleanVar() 
        self.make_R_at_H_bool.set(False)
        self.Lmake_R_at_H_bool = Checkbutton(frm_data_processor_2, text='Get R at desired field',
                                             var = self.make_R_at_H_bool) 
        self.Lmake_R_at_H_bool.pack(side=TOP, anchor = NW)
        
        ###Group By Index
        self.group_by_index_bool = BooleanVar() 
        self.group_by_index_bool.set(False)
        self.Lgroup_by_index_bool = Checkbutton(frm_data_processor_2, text='Group by Index',
                                                var = self.group_by_index_bool) 
        self.Lgroup_by_index_bool.pack(side=TOP, anchor = NW)
        
        
        #frm_data_processor_2.pack(side=LEFT)
        frm_data_processor_2.pack(expand = True, side = LEFT)
        frm_data_processor_3 = Frame(tab_1)
        
        self.Locsc = Label(frm_data_processor_3, text = 'Legend:\nOC = Open '+                           'Circuit\nSC = Short Circuit')
        self.Locsc.pack(side = TOP, fill = 'both', pady = 20)
        
        ###Submit and run routine
        self.number=Button(frm_data_processor_3, text = '!SUBMIT AND RUN ROUTINE!',
                           fg = 'black', command = self.call_routine)
        self.number.pack(side = TOP, fill = 'both')
        
        ###Exit program
        self.close=Button(frm_data_processor_3, text = 'Quit', fg = 'red', command = close_window)
        self.close.pack(side = TOP, fill = 'both')

        #frm_data_processor_3.pack(side=LEFT)
        frm_data_processor_3.pack(expand = True, side = LEFT)

        
        
        
        
        
        
        tab_2 = Frame(tabControl)
        tabControl.add(tab_2, text = 'Sensor Classifier')
        
        ###SENSOR CLASSIFIER
        frm_sensor_classifier_0 = Frame(tab_2)
 
        self.csvs_directory = None
        self.save_directory_predictions = None
        self.model = None
        
        ###Select CSVs directory
        self.csvs = Button(frm_sensor_classifier_0, text='Select CSVs Directory',
                           fg='black', command=self.call_csvs_names)
        self.csvs.pack(side=TOP, fill='both')
        #self.csvs_directory directory
        
        ###Select saving directory
        self.saving = Button(frm_sensor_classifier_0, text='Select Saving Directory',
                             fg='black', command=self.call_save_directory_predictions)
        self.saving.pack(side=TOP, fill='both')
        #save_directory_predictions directory
        
        ###Select Model directory
        self.model_ask = Button(frm_sensor_classifier_0, text='Select Model Directory',
                                fg='black', command=self.call_model)
        self.model_ask.pack(side=TOP, fill='both')
        #self.csvs_directory directory
        
        frm_sensor_classifier_0.pack(expand = True, side = LEFT)
        frm_sensor_classifier_1 = Frame(tab_2)
        
        ### Introduce saving name
        self.lsaving_name_predi = Label(frm_sensor_classifier_1, text='.xlsx filename:')
        self.lsaving_name_predi.pack(side=TOP, fill='both')
        self.saving_name_predi = Entry(frm_sensor_classifier_1)
        self.saving_name_predi.pack(side=TOP, fill='both')

        ###Progress Classification
        self.lprogress_bar_classi = Label(frm_sensor_classifier_1, text='Current Step:')
        self.lprogress_bar_classi.pack(side=TOP, fill='both')
        
        self.progress_text_classi = StringVar('') 
        self.progress_text_classi_field = Entry(frm_sensor_classifier_1,
                                                textvariable = self.progress_text_classi)
        self.progress_text_classi_field.pack(side = TOP, fill = 'both') 
        
        self.progress_bar_classi = ttk.Progressbar(frm_sensor_classifier_1,
                                                   orient = 'horizontal',
                                                   length = 5, mode = 'determinate')
        self.progress_bar_classi.pack(side = TOP, fill = 'both')

        frm_sensor_classifier_1.pack(expand = True, side = LEFT)
        frm_sensor_classifier_2 = Frame(tab_2)
        
        ###Submit and run routine
        self.call_routine_predi=Button(frm_sensor_classifier_2,
                                       text = '!SUBMIT AND MAKE PREDICTIONS!',
                                       fg = 'black',
                                       command = self.call_routine_predictions)
        self.call_routine_predi.pack(side = TOP, fill = 'both')
        
        ###Exit program
        self.close=Button(frm_sensor_classifier_2, text = 'Quit',
                          fg = 'red', command = close_window)
        self.close.pack(side = TOP, fill = 'both')

        #frm_data_processor_3.pack(side=LEFT)
        frm_sensor_classifier_2.pack(expand = True, side = LEFT)
        
        
            
            
            
            
            
            
        tab_3 = Frame(tabControl)
        tabControl.add(tab_3, text = 'Data Visualizer')
        
        
        ###DATA Visualizer
        frm_data_visualizer_1 = Frame(tab_3)

        self.meas_fold_graph = None
        self.maps_fold_graph = None
        self.saving_dir_graph = None


        ### Names for legend
        self.llegends = Label(frm_data_visualizer_1, text='Names for the legend:')
        self.llegends.pack(side=TOP, fill='both')
        self.legends = Entry(frm_data_visualizer_1)
        self.legends.pack(side=TOP, fill='both')
        
        ### Relative coordinates
        self.lrelative = Label(frm_data_visualizer_1, text='Relative coordinates:')
        self.lrelative.pack(side=TOP, fill='both')
        self.relative = Entry(frm_data_visualizer_1)
        self.relative.pack(side=TOP, fill='both')        
        
        ### Plot title
        self.ltitle = Label(frm_data_visualizer_1, text='Title for the plot:')
        self.ltitle.pack(side=TOP, fill='both')
        self.title = Entry(frm_data_visualizer_1)
        self.title.pack(side=TOP, fill='both')
        
        ### Introduce saving name
        self.lsaving_names = Label(frm_data_visualizer_1, text='.HTML filename:')
        self.lsaving_names.pack(side=TOP, fill='both')
        self.saving_names = Entry(frm_data_visualizer_1)
        self.saving_names.pack(side=TOP, fill='both')
        
        frm_data_visualizer_1.pack(expand = True, fill = X, side = LEFT)
        frm_data_visualizer_2 = Frame(tab_3)
        
        ###Select folder with measurement files
        self.meas_fold_b = Button(frm_data_visualizer_2, text='Select Measurements Directory',
                                  fg='black',
                                  command=self.call_meas_fold_graph)
        self.meas_fold_b.pack(side=TOP, fill='both')
        
        ###Select folder with maps files
        self.maps_fold_b = Button(frm_data_visualizer_2, text='Select Maps Directory',
                                  fg='black', 
                                  command=self.call_maps_fold_graph)
        self.maps_fold_b.pack(side=TOP, fill='both')
        
        ### Select saving folder
        self.saving_fold_b = Button(frm_data_visualizer_2, text = 'Select Saving Directory',
                                    fg = 'black',
                                    command = self.call_saving_dir_graph)
        self.saving_fold_b.pack(side = TOP, fill = 'both')
        
            
        ### Menu selections
        self.selection_variable = StringVar(frm_data_visualizer_2)
        self.selection_variable.set('Magnetoresistance') # default value

        self.lselection_variable = Label(frm_data_visualizer_2, text = 'Select variable to visualize:')
        self.lselection_variable.pack(side = TOP, fill = 'both')
        self.w = OptionMenu(frm_data_visualizer_2, self.selection_variable, 'Magnetoresistance',
                            'Resistance at desired field', 'Minimum Resistance', 'Predictions')
        self.w.pack(side=TOP, fill='both')

        ###Plot dies
        self.plot_die_bool = BooleanVar() 
        self.plot_die_bool.set(False)
        self.Lplot_die_bool = Checkbutton(frm_data_visualizer_2, text = 'Drawn Dies Frames',
                                          var = self.plot_die_bool) 
        self.Lplot_die_bool.pack(side=TOP, fill='both')
        
        frm_data_visualizer_2.pack(expand = True, fill = X, side = LEFT)
        frm_data_visualizer_3 = Frame(tab_3)
        
        ###Submit and run routine
        self.number=Button(frm_data_visualizer_3, text = '!SUBMIT AND RUN ROUTINE!', fg='black',
                           command = self.call_routine_graph)
        self.number.pack(side=TOP, fill='both')
        
        ###Exit program
        self.close=Button(frm_data_visualizer_3, text = 'Quit', fg = 'red', command = close_window)
        self.close.pack(side=TOP, fill='both')
        
        frm_data_visualizer_3.pack(expand = True, fill = X, side = LEFT)
        
        tabControl.pack(expan = 1, fill = 'both')
        
        
        
        
        
        
        
    ### Data processor   
        
    def call_smp_names(self):
        self.filesnames = filedialog.askopenfilenames(title = 'Select SMP files',
                                                      filetypes = (('SMP Files', '*.smp'),))
        messagebox.showinfo(title = 'INFO', message = 'SMP files selected: ' +                            str(self.filesnames))
        
    def call_save_directory(self):
        self.save_directory = filedialog.askdirectory(title = 'Select Saving Directory')
        messagebox.showinfo(title = 'INFO', message = 'Saving directory selected: ' +                            str(self.save_directory))

    def call_routine(self):
        stop = check_directories(self)
        if stop == 0:
            smps_directory = self.filesnames
            saving_directory = self.save_directory
            calib_H = float(self.calib_H.get())
            write_xslx_in = 1
            
            R_max_oc = float(self.High_R.get())
            R_min_sc = float(self.Low_R.get())
            
            # Automatic names
            files_names = []
            saving_names = []
            for i in smps_directory:
                files_names.append(i)
                prov = i[:-4]
                saving_names.append(prov.rsplit('/', 1)[-1])
            # End of automatic names
            self.progress_bar_smp['maximum'] = len(files_names)
            for i in range(len(files_names)):
                self.progress_bar_smp['value'] = i
                self.progress_bar_smp.update()
                file_name = files_names[i]
                saving_name = saving_names[i]
                df, num_sensors = process_data(file_name, calib_H,
                                               map_index_name = str(self.map_ind_n.get()),
                                               field_current_name = str(self.field_c_ind_n.get()),
                                               probes_current_name = str(self.probes_c_ind_n.get()),
                                               probes_voltage_name = str(self.probes_v_ind_n.get()))
                if self.make_RH_bool.get() == True:
                    make_RH(num_sensors, df, saving_name, saving_directory, self)
                if self.make_MR_bool.get() == True:
                    make_MR(num_sensors, df, saving_name, saving_directory, self,
                            write_xslx = write_xslx_in, threshold_R_oc = R_max_oc,
                            threshold_R_sc = R_min_sc)
                if self.make_Rmin_bool.get() == True:
                    make_Rmin(num_sensors, df, saving_name, saving_directory, self,
                              threshold_R_oc = R_max_oc, threshold_R_sc = R_min_sc)
                if self.make_R_at_H_bool.get() == True:
                    R_at_H = simpledialog.askfloat('R_at_H', 'R at H=')
                    make_R_at_H(num_sensors, df, saving_name, saving_directory, self, H = R_at_H,
                                write_xslx = write_xslx_in, threshold_R_oc = R_max_oc,
                                threshold_R_sc = R_min_sc)
            if self.group_by_index_bool.get() == True and self.make_RH_bool.get() == True: #Only available if RH curves are first done
                group_by_index(self)
            elif self.group_by_index_bool.get() == True and self.make_RH_bool.get() == False:
                messagebox.showwarning(title = 'WARNING',
                                       message = 'User tried to group by index without' +\
                                       ' making R(H) plots.')

            self.filesnames = None
            self.save_directory = None
            messagebox.showinfo(title = 'INFO', message = 'Computations completed! Check' +                                ' the saving folder.')
        self.progress_bar_smp['value'] = 0
        self.progress_text_inter.set('')

        
        
        
        
        
    ##Classifier
    
    def call_csvs_names(self):
        self.csvs_directory = filedialog.askdirectory(title = 'Select CSVs Directory')
        messagebox.showinfo(title = 'INFO', message = 'CSV directory selected: ' +                            str(self.csvs_directory))
    
    def call_save_directory_predictions(self):
        self.save_directory_predictions = filedialog.askdirectory(title = 'Select '+                                                                  'Saving Directory for predictions')
        messagebox.showinfo(title = 'INFO', message = 'Saving directory selected: ' +                            str(self.save_directory_predictions))
        
    def call_model(self):
        self.model = filedialog.askdirectory(title = 'Select Directory with model')
        messagebox.showinfo(title = 'INFO', message = 'Model folder selected: ' + str(self.model))
        
    def call_routine_predictions(self):
        stop = check_directories_predictions(self)
        
        if self.saving_name_predi.get() == '':
            messagebox.showwarning(title = 'WARNING',
                                   message = 'Since you did not give a filename for the excel' +\
                                   ' with predictions, it will be named ".xlsx".')
        
        if stop == 0:
            csvs_directory = self.csvs_directory
            saving_directory = self.save_directory_predictions
            model_directory = self.model
            predi_excel_name = self.saving_name_predi.get()
            
            #routine for predictions
            make_predictions_from_path(csvs_directory, model_directory, saving_directory,
                                       predi_excel_name, self)
            
            #Erasing data introduced by user so that no errors are commited on next prediction
            self.csvs_directory = None
            self.save_directory_predictions = None
            self.model = None
            
            messagebox.showinfo(title = 'INFO', message = 'Predictions completed! Check' +                                ' the saving folder.')
        
        
        
    ##Visualizer
    
    def call_meas_fold_graph(self):
        self.meas_fold_graph = filedialog.askdirectory(title = 'Select Directory with the' +                                                       ' Measurement Files')
        messagebox.showinfo(title = 'INFO', message = 'Measurement files directory ' +                            'selected: ' + str(self.meas_fold_graph))
        
    def call_maps_fold_graph(self):
        self.maps_fold_graph = filedialog.askdirectory(title = 'Select Directory with' +                                                       ' the Maps used in the Autoprober')
        messagebox.showinfo(title = 'INFO', message = 'Maps directory selected: ' +                            str(self.maps_fold_graph))
        
    def call_saving_dir_graph(self):
        self.saving_dir_graph = filedialog.askdirectory(title = 'Select Directory to ' +                                                        'save .HTML file')
        messagebox.showinfo(title = 'INFO', message = '.HTML file will be saved ' +                            'in: ' + str(self.saving_dir_graph))
    
    def call_routine_graph(self):
        stop = 0
        plot_title = self.title.get()
        files_measurements_dir = self.meas_fold_graph
        files_maps_dir = self.maps_fold_graph
        names = self.legends.get()
        path_saving_graph = self.saving_dir_graph

        
        if files_measurements_dir == '':
            files_measurements_dir = None
        if files_maps_dir == '':
            files_maps_dir = None
        if path_saving_graph == '':
            path_saving_graph = None

        if files_measurements_dir != None and files_maps_dir != None and path_saving_graph == None:
            messagebox.showerror(title='ERROR',
                                 message='No Saving Directory was selected, computations' +\
                                 ' interrupted!')
            stop = 1
        elif files_measurements_dir == None and files_maps_dir != None and path_saving_graph != None:
            messagebox.showerror(title='ERROR',
                                 message='No Measurements Directory was selected, computations'+\
                                 ' interrupted!')
            stop = 1
        elif files_measurements_dir != None and files_maps_dir == None and path_saving_graph != None:
            messagebox.showerror(title='ERROR',
                                 message='No Maps Directory was selected, computations'+\
                                 ' interrupted!')
            stop = 1     
        elif files_measurements_dir == None and files_maps_dir == None and path_saving_graph != None:
            messagebox.showerror(title='ERROR',
                                 message='No Measurements and Maps Directory were selected, '+\
                                 'computations interrupted!')
            stop = 1
        elif files_measurements_dir != None and files_maps_dir == None and path_saving_graph == None:
            messagebox.showerror(title='ERROR',
                                 message='No Maps and Saving Directory were selected, '+\
                                 'computations interrupted!')
            stop = 1
        elif files_measurements_dir == None and files_maps_dir != None and path_saving_graph == None:
            messagebox.showerror(title='ERROR',
                                 message='No Measurements and Saving Directory were selected, '+\
                                 'computations interrupted!')
            stop = 1    
        elif files_measurements_dir == None and files_maps_dir == None and path_saving_graph == None:
            messagebox.showerror(title='ERROR',
                                 message='No Measurements, Maps and Saving Directory were selected,' +\
                                 ' computations interrupted!')
            stop = 1
        elif files_measurements_dir != None and files_maps_dir != None and path_saving_graph != None:
            stop = 0

        
        
        str_all_coordinate = self.relative.get()
        count = 0
        for i in str_all_coordinate:
            if i == '(':
                count += 1
        if count == 0:
            stop = 1
            messagebox.showerror(title = 'ERROR',
                                   message = 'You need to give at least one pair of relative' +\
                                 ' coordinates even if it is "(0, 0)".')
            list_all_coordinate = []
        elif count == 1:
            list_all_coordinate = []
            list_all_coordinate.append(eval(str_all_coordinate))
        else:
            list_all_coordinate = list(ast.literal_eval(str_all_coordinate))
        
        if names == '':
            messagebox.showwarning(title = 'WARNING',
                                   message = 'Since you did not provide names for datasets the' +\
                                   ' legend will be empty.')
        names = names.split(', ')
        
        if self.saving_names.get() == '':
            messagebox.showwarning(title = 'WARNING',
                                   message = 'Since you did not give a filename for the .HTML' +\
                                   ' it will be named ".html".')
        if plot_title == '':
            messagebox.showwarning(title = 'WARNING',
                                   message = 'Since you did not provide a title for the ' +\
                                   'plot it will be empty.')
        
        if files_maps_dir != None:
            files_maps = get_all_name(files_maps_dir, '.xlsx')
        else:
            files_maps = []
        
        selected_variable = self.selection_variable.get()
        
        selection_type = None
        if selected_variable == 'Magnetoresistance' and stop == 0:
            quantity = 'MR [%]'
            selection_type = 'Number'
            files_measurements = get_all_name(files_measurements_dir, '.xlsx')
        elif selected_variable == 'Resistance at desired field' and stop == 0:
            quantity = 'Resistance [Ω]'
            selection_type = 'Number'
            files_measurements = get_all_name(files_measurements_dir, 'avgs.xlsx')
        elif selected_variable == 'Minimum Resistance' and stop == 0:
            quantity = 'Rmin [Ω]'
            selection_type = 'Number'
            files_measurements = get_all_name(files_measurements_dir, '.xlsx')
        elif selected_variable == 'Predictions' and stop == 0:
            quantity = 'Prediction'
            selection_type = 'String'
            files_measurements = get_all_name(files_measurements_dir, '.xlsx')
            
        if stop == 0:
            if len(names) != len(files_measurements) or len(names) != len(files_maps) or +                len(names) != len(list_all_coordinate):
                stop = 1
                messagebox.showerror(title='ERROR', message='Check the ammount of data files,' +                                     ' maps, legend and relative coordinates, they should' +                                     ' be the same!' +                                     '\nNumber of Measurement files given: ' + str(len(files_measurements)) +                                     '\nNumber of Map files given: ' + str(len(files_maps)) +                                     '\nNumber of Legend Names given: ' + str(len(names)) +                                     '\nNumber of relative coordinates given: ' + str(len(list_all_coordinate)))
        
        if stop == 0:
            html_name = path_saving_graph + '/' + self.saving_names.get() + '.html'
            fig = go.Figure()
            if self.plot_die_bool.get() == True:
                self.n_dies_x = simpledialog.askinteger('n_dies_x',
                                                        'Enter the number of dies in the x direction:')
                self.n_dies_y = simpledialog.askinteger('n_dies_y',
                                                        'Enter the number of dies in the y direction:')
                self.dies_x_dim = simpledialog.askfloat('dies_x_dimension',
                                                              'Enter the dimension of the die in the' +\
                                                        ' x direction:')
                self.dies_y_dim = simpledialog.askfloat('dies_y_dimension',
                                                              'Enter the dimension of the die in the' +\
                                                        ' y direction:')
                if self.n_dies_x != None and self.n_dies_y != None and +                    self.dies_x_dim != None and self.dies_y_dim != None:
                    plot_die(self.n_dies_x, self.n_dies_y, 0, 0, self.dies_x_dim, self.dies_y_dim, fig)
                else:
                    messagebox.showwarning(title = 'WARNING', message = 'Rectangular dies will ' +                                           'not be plotted because you did not give all the ' +                                           'needed parameters!')
                self.n_dies_x = None
                self.n_dies_y = None
                self.dies_x_dim = None
                self.dies_y_dim = None
                
            files_meas_print_g = []
            for i in files_measurements:
                files_meas_print_g.append(i.split('/')[-1])
            
            files_maps_print_g = []
            for i in files_maps:
                files_maps_print_g.append(i.split('/')[-1])
                
            order_names = messagebox.askquestion(title = 'WARNING', message = 'Are all the files below in the same order?' +
                                                 '\nOrder of Names: ' + str(names)[1:-1] +
                                                 '\nOrder of Relative Coordinates: ' + str(list_all_coordinate)[1:-1] +
                                                 '\nOrder of Measurement files: ' + str(files_meas_print_g)[1:-1] +
                                                 '\nOrder of Map files: ' + str(files_maps_print_g)[1:-1]
                                                 , icon = 'warning')
            
            if order_names == 'yes' and selection_type == 'Number':
                all_maps_list = []
                all_data_list = []
                for i in range(len(files_maps)):
                    all_maps_list.append(pd.read_excel(files_maps[i]))
                    all_data_list.append(pd.read_excel(files_measurements[i]))

                list_rele_df = get_rele_data(all_maps_list, all_data_list, 0, None, quantity)

                dfs = change_coordinates_put_name(list_all_coordinate, list_rele_df, names)

                all_data = pd.concat(dfs)
                data_good = all_data[all_data[quantity].apply(lambda x: isinstance(x, float))]

                max_color = myround(data_good[quantity].max(), base=10)
                min_color = 0


                # Add traces
                for i in range(len(names)):
                    df = dfs[i]
                    df_floats = df[df[quantity].apply(lambda x: isinstance(x, float))]
                    data_SC = df[df[quantity] == 'SC']
                    data_OC = df[df[quantity] == 'OC']

                    fig.add_trace(go.Scatter(x=df_floats['x abs'], y=df_floats['y abs'],
                            marker=dict(size=7.5, color = df_floats[quantity], coloraxis = 'coloraxis'),
                            mode='markers', name =  df_floats['quantity'].iloc[0],
                            text = df_floats['Map Index'],
                            hovertemplate = '<br><b>X</b>: %{x}' + 
                                            '<br><b>Y</b>: %{y}' +
                                            '<br><b>Map Index</b>: %{text}<br><b>' + 
                                            str(quantity) + '</b>: %{marker.color:.1f}', ))

                    fig.add_trace(go.Scatter(x=data_SC['x abs'], y=data_SC['y abs'],
                            marker=dict(size=7.5, color = 'purple'),
                            mode='markers', name = df_floats['quantity'].iloc[0] + ' SC', 
                            text = data_SC['Map Index'],
                            hovertemplate = '<br><b>X</b>: %{x}' +
                                            '<br><b>Y</b>: %{y}' +
                                            '<br><b>Map Index</b>: %{text}<br><b>' +
                                            str(quantity) + '</b>: Shorted', ))

                    fig.add_trace(go.Scatter(x=data_OC['x abs'], y=data_OC['y abs'],
                             marker=dict(size=7.5, color = 'red'),
                             mode='markers', name = df_floats['quantity'].iloc[0] + ' OC',
                             text = data_OC['Map Index'],
                             hovertemplate = '<br><b>X</b>: %{x}' +
                                             '<br><b>Y</b>: %{y}' +
                                             '<br><b>Map Index</b>: %{text}<br><b>' +
                                             str(quantity) + '</b>: Open', ))


                fig.update_xaxes(title_text = 'x - Absolute Position [mm]', showline = True, linewidth = 1,
                                 linecolor = 'black', mirror = True, gridcolor = 'LightGrey')
                fig.update_yaxes(title_text = 'y - Absolute Position [mm]', showline = True, linewidth = 1,
                                 linecolor = 'black', mirror = True, gridcolor = 'LightGrey')
                fig.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 1),
                                  plot_bgcolor = 'rgb(255,255,255)',
                                  coloraxis = {'colorscale':'Inferno',
                                               'colorscale' : [[0, 'green'],
                                                              [0.5, 'rgb(255, 192, 0)'],
                                                              [1.0, 'rgb(255, 0, 0)']],
                                               'cmin' : min_color, 'cmax' : max_color},
                                  coloraxis_colorbar=dict(title = quantity, ticks='outside'),
                                  legend=dict(x = -0.3, y = 0.5, title = 'Select Quantities:'),
                                  title = plot_title)
                fig.write_html(html_name)

                self.meas_fold_graph = None
                self.maps_fold_graph = None
                self.saving_dir_graph = None
                selection_type = None
                
                messagebox.showinfo(title = 'INFO', message = 'Plot completed! Check the saving folder.')
                

            if order_names == 'yes' and selection_type == 'String':
                all_maps_list = []
                all_data_list = []
                for i in range(len(files_maps)):
                    all_maps_list.append(pd.read_excel(files_maps[i]))
                    all_data_list.append(pd.read_excel(files_measurements[i]))

                list_rele_df = get_rele_data_predi(all_maps_list, all_data_list, 0, None)

                dfs = change_coordinates_put_name(list_all_coordinate, list_rele_df, names)

                all_data = pd.concat(dfs)

                # Add traces
                for i in range(len(names)):
                    df = dfs[i]
                    data_OK = df[df[quantity] == 'OK']
                    data_M = df[df[quantity] == 'M']
                    data_A = df[df[quantity] == 'A']
                    data_NOK = df[df[quantity] == 'NOK']
                    
                    if not data_OK.empty:
                        fig.add_trace(go.Scatter(x=data_OK['x abs'], y=data_OK['y abs'],
                                marker=dict(size=7.5, color = 'green'), mode='markers',
                                name = data_OK['quantity'].iloc[0] + ' OK',
                                customdata = np.stack((data_OK['Map Index'], data_OK['OK'],
                                                       data_OK['M'], data_OK['A'], data_OK['NOK']), axis=-1),
                                hovertemplate = ('<br><b>X</b>: %{x}'+\
                                                 '<br><b>Y</b>: %{y}'+\
                                                 '<br><b>Map Index</b>: %{customdata[0]}'+\
                                                 '<br><b>Probability OK</b>: %{customdata[1]:.3f}'+\
                                                 '<br><b>Probability M</b>: %{customdata[2]:.3f}'+\
                                                 '<br><b>Probability A</b>: %{customdata[3]:.3f}'+\
                                                 '<br><b>Probability NOK</b>: %{customdata[4]:.3f}<br>')))
                    
                    if not data_M.empty:
                        fig.add_trace(go.Scatter(x=data_M['x abs'], y=data_M['y abs'],
                                marker=dict(size=7.5, color = 'yellow'), mode='markers',
                                name = data_M['quantity'].iloc[0] + ' M',
                                customdata = np.stack((data_M['Map Index'], data_M['OK'],
                                                       data_M['M'], data_M['A'], data_M['NOK']), axis=-1),
                                hovertemplate = ('<br><b>X</b>: %{x}'+\
                                                 '<br><b>Y</b>: %{y}'+\
                                                 '<br><b>Map Index</b>: %{customdata[0]}'+\
                                                 '<br><b>Probability OK</b>: %{customdata[1]:.3f}'+\
                                                 '<br><b>Probability M</b>: %{customdata[2]:.3f}'+\
                                                 '<br><b>Probability A</b>: %{customdata[3]:.3f}'+\
                                                 '<br><b>Probability NOK</b>: %{customdata[4]:.3f}<br>')))
                    if not data_A.empty:
                        fig.add_trace(go.Scatter(x=data_A['x abs'], y=data_A['y abs'],
                                marker=dict(size=7.5, color = 'orange'), mode='markers',
                                name = data_A['quantity'].iloc[0] + ' A',
                                customdata = np.stack((data_A['Map Index'], data_A['OK'],
                                                       data_A['M'], data_A['A'], data_A['NOK']), axis=-1),
                                hovertemplate = ('<br><b>X</b>: %{x}'+\
                                                 '<br><b>Y</b>: %{y}'+\
                                                 '<br><b>Map Index</b>: %{customdata[0]}'+\
                                                 '<br><b>Probability OK</b>: %{customdata[1]:.3f}'+\
                                                 '<br><b>Probability M</b>: %{customdata[2]:.3f}'+\
                                                 '<br><b>Probability A</b>: %{customdata[3]:.3f}'+\
                                                 '<br><b>Probability NOK</b>: %{customdata[4]:.3f}<br>')))
                    if not data_NOK.empty:
                        fig.add_trace(go.Scatter(x=data_NOK['x abs'], y=data_NOK['y abs'],
                                marker=dict(size=7.5, color = 'red'), mode='markers',
                                name = data_NOK['quantity'].iloc[0] + ' NOK',
                                customdata = np.stack((data_NOK['Map Index'], data_NOK['OK'],
                                                       data_NOK['M'], data_NOK['A'], data_NOK['NOK']), axis=-1),
                                hovertemplate = ('<br><b>X</b>: %{x}'+\
                                                 '<br><b>Y</b>: %{y}'+\
                                                 '<br><b>Map Index</b>: %{customdata[0]}'+\
                                                 '<br><b>Probability OK</b>: %{customdata[1]:.3f}'+\
                                                 '<br><b>Probability M</b>: %{customdata[2]:.3f}'+\
                                                 '<br><b>Probability A</b>: %{customdata[3]:.3f}'+\
                                                 '<br><b>Probability NOK</b>: %{customdata[4]:.3f}<br>')))


                fig.update_xaxes(title_text = 'x - Absolute Position [mm]', showline = True, linewidth = 1,
                                 linecolor = 'black', mirror = True, gridcolor = 'LightGrey')
                fig.update_yaxes(title_text = 'y - Absolute Position [mm]', showline=True, linewidth=1,
                                 linecolor='black', mirror=True, gridcolor='LightGrey')
                
                fig.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 1),
                                  plot_bgcolor = 'rgb(255,255,255)',
                                  legend = dict(x = -0.3, y = 0.5, title = 'Select Quantities:'),
                                  title = plot_title)

                fig.write_html(html_name)

                self.meas_fold_graph = None
                self.maps_fold_graph = None
                self.saving_dir_graph = None
                selection_type = None
                
                messagebox.showinfo(title = 'INFO', message = 'Plot completed! Check the saving folder.')


root = Tk() # create window
root.configure(bg='lightgrey')
root.title('AutoProber Data Processor, Sensor Classifier & Visualizer v.4.0')
root.wm_iconbitmap('INESC-MN_logo.ico')


app = App(root)
root.mainloop()

