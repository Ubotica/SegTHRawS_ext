"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import re
import onnx
import torch
import shutil
import warnings
import onnxruntime
import numpy as np
import pandas as pd
import lightning as pl
import matplotlib.pyplot as plt

from train_constants import DEFAULT_SEED, OPSET_VERSION


warnings.filterwarnings(category=FutureWarning,action='ignore')
warnings.filterwarnings(category=torch.jit.TracerWarning,action='ignore')


##### MODEL CONVERSION TO ONNX
def convert_model_to_onnx(model: pl.LightningModule,
                          model_checkpoint_path: str,
                          input_shape: tuple = (1,3,256,256),
                          opset_version: int = OPSET_VERSION,
                          seed: int = DEFAULT_SEED) -> None:
    
    torch.manual_seed(seed)

    # trained_model = ThermalModel.load_from_checkpoint(checkpoint_path=model_checkpoint_path,model=model,loss_fn=loss)

    onnx_path = os.path.join(os.path.dirname(model_checkpoint_path),'MYRIAD','ONNX')
    os.makedirs(onnx_path,exist_ok=True)

    if len(input_shape) == 3:
        input_shape = input_shape.unsqueeze(0)
    elif len(input_shape) !=4 and len(input_shape) !=3:
        raise ValueError('Shape mismatch: The input shape must have ndim=3 or ndim=4.')
    
    x = torch.randn(input_shape).cpu()
    model_onnx = model.cpu()
    model_onnx.eval()
    torch_out = model_onnx(x)

    onnx_model_path = os.path.join(onnx_path,os.path.basename(model_checkpoint_path) + '.onnx')

    # Export the model
    torch.onnx.export(model_onnx,                   # model being run
                      x,                            # model input (or a tuple for multiple inputs)
                      onnx_model_path,              # where to save the model (can be a file or file-like object)
                      export_params=True,           # store the trained parameter weights inside the model file
                      opset_version=opset_version,  # the ONNX version to export the model. 16 is not supported by Myriad X. 
                      do_constant_folding=True,     # whether to execute constant folding for optimization
                      input_names = ['input'],      # the model's input names
                      output_names = ['output'],    # the model's output names
                      )
    ## Checking that the model is correct
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    ## Check that the ONNX model behave as expected
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    try:
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been successfully tested with ONNXRuntime.")
    except:

        print("The ONNX model has not been converted successfully")
        with open(os.path.join(onnx_path,'onnx_conversion_ERROR.txt','w')) as file:
            pass

###### PLOT GENERATION

def generate_metrics_plots(model_main_path: str,
                           model_seed_path: str,
                           model_checkpoint_path: str,
                           metrics_path: str) -> None:


    metrics_version_folder_path = os.path.join(model_main_path,'csv_logs',f'{os.path.basename(model_seed_path)}')

    version_files = os.listdir(metrics_version_folder_path)

    version_files.sort(key=lambda x: os.path.getmtime(os.path.join(metrics_version_folder_path,x)),reverse=True)

    metrics_df = pd.read_csv(os.path.join(metrics_version_folder_path,version_files[0],'metrics.csv'))

    df = metrics_df.copy()

    selected_rows = df.dropna(subset=['epoch'],axis=0)
    input_epoch = df['epoch'].max()

    #Move the test values from epoch 0 to the epoch of the model. Select any of the test metrics
    for column in df.columns:
        if column.startswith('test_'):
            selected_rows.loc[df[column].notna(),'epoch'] = input_epoch
            break

    combined_df = pd.DataFrame(columns=selected_rows.columns)

    for _, group in selected_rows.groupby('epoch'):
        if len(group)==2:

            row1 = group.iloc[0]
            row2 = group.iloc[1]

            
            combined_row = pd.DataFrame({
                col: [row1[col] if pd.notna(row1[col]) else row2[col]] for col in df.columns
            })

            combined_df = pd.concat([combined_df,combined_row],ignore_index=True)
        elif len(group)==3:
            row1 = group.iloc[0]
            row2 = group.iloc[1]
            row3 = group.iloc[2]

            
            combined_row = pd.DataFrame({
                col: [row1[col] if pd.notna(row1[col]) else row2[col] if pd.notna(row2[col]) else row3[col]] for col in df.columns
            })

            combined_df = pd.concat([combined_df,combined_row],ignore_index=True)

    csv_name = os.path.basename(model_checkpoint_path)+ '.csv'

    metrics_csv_path = os.path.join(metrics_path,f'metrics_{csv_name}')

    plots_path = os.path.join(os.path.dirname(metrics_csv_path),'plots')
    os.makedirs(plots_path,exist_ok=True)

    combined_df.to_csv(metrics_csv_path,index=False)


    #Generate learning rate evolution plot

    fig, ax = plt.subplots()

    ax.plot(metrics_df['lr-Adam'].dropna(),color='g')
    ax.set_xlabel('# steps')
    ax.set_ylabel(f'Lr evolution')
    ax.set_title(f'Lr evolution',fontname="Charter",weight='bold')

    plt.yscale('log')
    plt.savefig(os.path.join(plots_path,'Lr_evolution.png'))

    # ax.plot(train_values,color='b',label=f'Validation {train_values[len(train_values)-1]:.3e}')
    # plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.3e}"))

    shutil.rmtree(os.path.join(metrics_version_folder_path,version_files[0])) #Delete the old metrics file
    shutil.rmtree(os.path.join(model_main_path,'csv_logs')) #Delete the old metrics file



    ### Generation of the rest of the metrics plots

    metric_id_list = []

    csv_file = pd.read_csv(os.path.join(metrics_csv_path))

    columns_to_drop = ['lr-Adam','step'] + [test_metric for test_metric in combined_df.columns if test_metric.startswith('test')]

    # columns_to_drop = ['lr-Adam','step','test_IoU','test_fbeta','test_loss','test_IoU_0.5','test_IoU_0.75','test_IoU_0.9']
    # columns_to_drop = ['lr-Adam','step','test_IoU','test_acc','test_fbeta','test_jaccard','test_loss','train_acc','val_acc']

    new_test_df = csv_file.drop(columns=columns_to_drop)

    for metric_name in new_test_df.columns[1:]:
        fig, ax = plt.subplots(figsize=(7,5))
        if metric_name[:5]=='train':
            try:
                ax.set_xlabel('# epochs')

                metric_id = metric_name.replace('train_','')    
                if metric_id not in metric_id_list:
                    val_metric_name = metric_name.replace('train','val')
                    train_values = new_test_df[metric_name].astype('float32')
                    val_values = new_test_df[val_metric_name].astype('float32')

                    if metric_id == 'loss':
                        ax.plot(train_values,color='r',label=f'Training {max(train_values):.3e}')
                        ax.plot(val_values,color='b',label=  f'Validation    {max(val_values):.3e}')
                    
                    else: #This is to see the final metric better of IoU and Fbeta
                        ax.plot(train_values,color='r',label=f'Training {max(train_values):.3f}')
                        ax.plot(val_values,color='b',label=  f'Validation    {max(val_values):.3f}')
                    
                    if metric_id == 'fbeta':
                        ax.set_title('F-1 evolution',fontname="Charter",weight='bold')
                    else:
                        ax.set_title(f'{metric_id[0].upper()+metric_id[1:]} evolution',fontname="Charter",weight='bold')
                    
                    ax.set_ylabel(f'{metric_id[0].upper()+metric_id[1:]}')
                    
                    # plt.annotate('%0.2f' % train_values[len(train_values)-1], xy=(len(train_values)-1, train_values[len(train_values)-1]))
                    # plt.annotate('%0.2E' % train_values[len(train_values)-1], xy=(new_test_df['epoch'][len(new_test_df['epoch'])-1], train_values[len(train_values)-1]),xytext=(-40,5),textcoords='offset pixels')
                    # plt.annotate('%0.2E' % val_values[len(val_values)-1], xy=(new_test_df['epoch'][len(new_test_df['epoch'])-1], val_values[len(val_values)-1]),xytext=(-40,5),textcoords='offset pixels')
                    metric_id_list.append(metric_id)
                else:
                    metric_id = None
                    plt.close(fig)
            except: #Case when a variable does not appear in validation. Designed for IoU_0.9
                plt.close(fig)
                continue

        elif metric_name[:3]=='val':
            
            try:

                ax.set_xlabel('# epochs')
                
                metric_id = metric_name.replace('val_','')
                if metric_id not in metric_id_list:

                    train_metric_name = metric_name.replace('val','train')
                    train_values = new_test_df[metric_name].astype('float32')
                    val_values = new_test_df[train_metric_name].astype('float32')

                    if metric_id == 'loss':
                        ax.plot(train_values,color='r',label=f'Training {max(train_values):.3e}')
                        ax.plot(val_values,color='b',label=  f'Validation    {max(val_values):.3e}')
                    
                    else: #This is to see the final metric better of IoU and Fbeta
                        ax.plot(train_values,color='r',label=f'Training {max(train_values):.3f}')
                        ax.plot(val_values,color='b',label=  f'Validation    {max(val_values):.3f}')


                    if metric_id == 'fbeta':
                        ax.set_title('F-1 evolution',fontname="Charter",weight='bold')
                    else:
                        ax.set_title(f'{metric_id[0].upper()+metric_id[1:]} evolution',fontname="Charter",weight='bold')
                    
                    ax.set_ylabel(f'{metric_id[0].upper()+metric_id[1:]}')

                    # plt.annotate('%0.2E' % train_values[len(train_values)-1], xy=(new_test_df['epoch'][len(new_test_df['epoch'])-1], train_values[len(train_values)-1]),xytext=(-40,5),textcoords='offset pixels')

                    # plt.annotate('%0.2E' % val_values[len(val_values)-1], xy=(new_test_df['epoch'][len(new_test_df['epoch'])-1], val_values[len(val_values)-1]),xytext=(-40,5),textcoords='offset pixels')
                    
                    metric_id_list.append(metric_id)
                else:
                    metric_id = None
                    plt.close(fig)
            except: #Case when a variable does not appear in training. Just for redundancy
                plt.close(fig)
                continue

        else: #This ensures that only the train and validation metrics plots  are saved
            plt.close(fig)

        if metric_id:
            ax.legend()
            plt.savefig(os.path.join(plots_path,metric_id+'_evolution.png'))
            # plt.show()
                
def generate_seed_metrics(model_seed_path : str):


    final_outputs = pd.DataFrame()

    seed = re.match(r"seed_(\d+)",os.path.basename(model_seed_path)).group(1)

    for run in os.listdir(model_seed_path):
        
        if run.startswith('run'):
            
            n_run = re.match(r'run_(\d+)',run).group(1)
            csv_path = [os.path.join(model_seed_path,run,'metrics',csv_file) for csv_file in os.listdir(os.path.join(model_seed_path,run,'metrics')) if csv_file.endswith('.csv') or csv_file.startswith('metrics_')][0]

            csv = pd.read_csv(csv_path)

            test_metrics_names_list = [column for column in csv.columns if column.startswith('test_')]
            test_metrics = (csv.loc[csv[csv['test_IoU_event'].notna()].index,test_metrics_names_list]).dropna().reset_index().drop('index',axis='columns')

            epoch = float(re.match(r"metrics_seed_(\d+)_epoch=(\d+)\.csv",os.path.basename(csv_path)).group(2)) #Group 2 will get the epoch number
            
            train_val_columns = [column for column in csv.columns if column.startswith('train') or column.startswith('val')]

            train_val_metrics = (csv.loc[csv[csv['epoch'] == epoch].index,train_val_columns]).dropna().reset_index().drop('index',axis='columns')

            results = pd.concat([test_metrics,train_val_metrics],axis=1)
            results_col = results.T
            results_col.columns = [f'run {n_run}']
            new_row_data = pd.DataFrame({f'run {n_run}': [epoch]})

            # Append the new row to the DataFrame and reset index
            results_col = pd.concat([new_row_data,results_col]).rename(index={0: 'Model epoch'})

            # Add the respective run to the final outputs
            final_outputs = pd.concat([final_outputs,results_col],axis=1)



    # Obtain the index of runs
    runs_numbers = [int(col.split(' ')[1]) for col in final_outputs.columns]

    # Find the maximum and minimum run numbers
    max_run = max(runs_numbers)
    min_run = min(runs_numbers)

    sorted_columns = ['run ' + str(i) for i in range(min_run, max_run + 1)]

    csv_save_path = os.path.join(model_seed_path,f'seed_{seed}_metrics.csv')


    final_outputs = final_outputs[sorted_columns]
    final_outputs.to_csv(csv_save_path)


def generate_batch_size_metrics(model_batch_size_path : str):

    batch_size_metrics_output = pd.DataFrame()

    batch_size = os.path.basename(model_batch_size_path)

    seeds_list = sorted([seed for seed in os.listdir(model_batch_size_path) if seed.startswith('seed')])

    for seed in seeds_list:
        if seed.startswith('seed'):
            model_seed_path = os.path.join(model_batch_size_path,seed)
            
            generate_seed_metrics(model_seed_path=model_seed_path)

            seed_metrics_csv = pd.read_csv(os.path.join(model_batch_size_path,seed,f'{seed}_metrics.csv'))
                    
            seed_metrics_csv.columns.values[0] = 'metrics'
            new_row_data = pd.DataFrame({'metrics': [seed]})

            # Append the new row to the DataFrame and reset index
            seed_metrics_csv = pd.concat([new_row_data,seed_metrics_csv]).reset_index().drop(['index'],axis='columns')
            # nan_column = pd.DataFrame({np.nan: [np.nan] * len(seed_metrics_csv)}, index=seed_metrics_csv.index)

            batch_size_metrics_output = pd.concat([batch_size_metrics_output,seed_metrics_csv]).reset_index().drop(['index'],axis='columns')


    batch_size_metrics_output.to_csv(os.path.join(model_batch_size_path,f'{batch_size}_metrics.csv'),index=False)


def generate_model_metrics(model_name_path : str):

    model_metrics_output = pd.DataFrame()
    model_name = os.path.basename(model_name_path)

    batch_sizes_list = sorted([bs for bs in os.listdir(model_name_path) if bs.startswith('batch_size')])

    for bs in batch_sizes_list:
        if bs.startswith('batch_size'):
            model_batch_size_path = os.path.join(model_name_path,bs)
            
            generate_batch_size_metrics(model_batch_size_path=model_batch_size_path)

            bs_metrics_csv = pd.read_csv(os.path.join(model_name_path,bs,f'{bs}_metrics.csv'))
                    
            bs_metrics_csv.columns.values[0] = 'metrics'
            new_row_data = pd.DataFrame({'metrics': [bs]})

            # Append the new row to the DataFrame and reset index
            bs_metrics_csv = pd.concat([new_row_data,bs_metrics_csv]).reset_index().drop(['index'],axis='columns')
            # nan_column = pd.DataFrame({np.nan: [np.nan] * len(seed_metrics_csv)}, index=seed_metrics_csv.index)

            model_metrics_output = pd.concat([model_metrics_output,bs_metrics_csv]).reset_index().drop(['index'],axis='columns')


    model_metrics_output.to_csv(os.path.join(model_name_path,f'{model_name}_metrics.csv'),index=False)
