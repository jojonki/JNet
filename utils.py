from datetime import datetime
import shutil
import torch


def save_checkpoint(state, is_best, filename='checkpoint.model', best_filename='model_best.model'):
    print('save_model', filename, best_filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def now():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def debug_log(text=''):
    msg = now()
    if text:
        msg += ': ' + text
    print(msg)
