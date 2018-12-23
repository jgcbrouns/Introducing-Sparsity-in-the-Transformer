import sys
import time
import datetime


def createHistoryFile(model_parameters, sys_argv):
    timestamp = int(time.time())
    date = datetime.datetime.fromtimestamp(timestamp)
    timestamp = date.strftime('%Y-%m-%d_%H:%M:%S')
    filepath = 'logs/'+'run_'+timestamp+'.csv'
    with open(filepath, 'a') as csv_file:

        if 'sparse' in sys_argv:
            csv_file.write('model_type, sparse' + '\n')
        if 'originalImplementation' in sys_argv:
            csv_file.write('model_type, originalImplementation' + '\n')
        if 'originalWithTransfer' in sys_argv:
            csv_file.write('model_type, originalWithTransfer' + '\n')

        if 'origdata' in sys_argv:
            csv_file.write('trainingset, origdata' + '\n')
        elif 'testdata' in sys_argv:
            csv_file.write('trainingset, testdata' + '\n')
        
        if 'load_existing_model' in sys_argv:
            csv_file.write('load_existing_model, true' + '\n')
        else:
            csv_file.write('load_existing_model, false' + '\n')

        for param_name,value in sorted(model_parameters.items()):
            csv_file.write(param_name + ', ' + str(value) + '\n')

        csv_file.write('\n')

        # csv_file.write('loss, val_accu, val_ppl, val_loss, accu, ppl,\n')
        csv_file.write('accu, loss, ppl, val_accu, val_loss, val_ppl,\n')

    return filepath

def writeEpochHistoryToDisk(history, filepath):
    with open(filepath, 'a') as csv_file:
        epoch_results_string = ''
        for key, value in sorted(history.history.items()):
            epoch_results_string = epoch_results_string + str(value[0]) + ','
        csv_file.write(epoch_results_string+'\n')


