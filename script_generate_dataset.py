import os
import numpy as np
import struct
from typing import Callable, Dict, Optional, Tuple
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import imageio


dvs_path = "/home/zhang/S2/RP/DataSet/DVS_Gesture_dataset/DvsGesture"
events_dvs_split_path = "/home/zhang/TestPath/DVS_npz_events"
frames_dvs_split_path = "/home/zhang/TestPath/DVS_npz_frames"
visualization_saved_path= ""


def load_aedat_v3(file_name: str) -> Dict:

    with open(file_name, 'rb') as bin_f:
        # skip ascii header
        line = bin_f.readline()
        while line.startswith(b'#'):
            if line == b'#!END-HEADER\r\n':
                break
            else:
                line = bin_f.readline()

        txyp = {
            't': [],
            'x': [],
            'y': [],
            'p': []
        }
        while True:
            header = bin_f.read(28)
            if not header or len(header) == 0:
                break

            # read header
            e_type = struct.unpack('H', header[0:2])[0]
            e_source = struct.unpack('H', header[2:4])[0]
            e_size = struct.unpack('I', header[4:8])[0]
            e_offset = struct.unpack('I', header[8:12])[0]
            e_tsoverflow = struct.unpack('I', header[12:16])[0]
            e_capacity = struct.unpack('I', header[16:20])[0]
            e_number = struct.unpack('I', header[20:24])[0]
            e_valid = struct.unpack('I', header[24:28])[0]

            data_length = e_capacity * e_size
            data = bin_f.read(data_length)
            counter = 0

            if e_type == 1:
                while data[counter:counter + e_size]:
                    aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                    timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                    x = (aer_data >> 17) & 0x00007FFF
                    y = (aer_data >> 2) & 0x00007FFF
                    pol = (aer_data >> 1) & 0x00000001
                    counter = counter + e_size
                    txyp['x'].append(x)
                    txyp['y'].append(y)
                    txyp['t'].append(timestamp)
                    txyp['p'].append(pol)
            else:
                # non-polarity event packet, not implemented
                pass
        txyp['x'] = np.asarray(txyp['x'])
        txyp['y'] = np.asarray(txyp['y'])
        txyp['t'] = np.asarray(txyp['t'])
        txyp['p'] = np.asarray(txyp['p'])
        return txyp
    


def split_aedat_files_to_npz(fname: str, aedat_file: str, csv_file: str, output_dir: str):
    events = load_aedat_v3(aedat_file)
    print(f'Start to split [{aedat_file}] to samples.')
    # read csv file and get time stamp and label of each sample
    # then split the origin data to samples
    csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

    # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
    file_num = [0] * 11

    for i in range(csv_data.shape[0]):
        # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
        label = csv_data[i][0] - 1
        t_start = csv_data[i][1]
        t_end = csv_data[i][2]
        mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
        output_file_name = os.path.join(output_dir, str(label), f'{fname}_{file_num[label]}.npz')
        np.savez(output_file_name,
                    t=events['t'][mask],
                    x=events['x'][mask],
                    y=events['y'][mask],
                    p=events['p'][mask]
                    )
        print(f'[{output_file_name}] saved.')
        file_num[label] += 1


def classify_aedat_to_npz(origin_dataset_path: str, events_npz_path: str):

    train_dir = os.path.join(events_npz_path, 'train')
    test_dir = os.path.join(events_npz_path, 'test')
    max_threads_number_for_datasets_preprocess = 16  #my computer has 16 threads 
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    print(f'Mkdir [{train_dir, test_dir}].')
    for label in range(11):
        os.mkdir(os.path.join(train_dir, str(label)))
        os.mkdir(os.path.join(test_dir, str(label)))
    print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

    with open(os.path.join(origin_dataset_path, 'trials_to_train.txt')) as trials_to_train_txt, open(
            os.path.join(origin_dataset_path, 'trials_to_test.txt')) as trials_to_test_txt:
        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), max_threads_number_for_datasets_preprocess)) as tpe:
            print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

            for fname in trials_to_train_txt.readlines():
                fname = fname.strip()
                if fname.__len__() > 0:  #we can see in the files some lines are empty
                    aedat_file = os.path.join(origin_dataset_path, fname)
                    fname = os.path.splitext(fname)[0] #remove .aedat
                    tpe.submit(split_aedat_files_to_npz, fname, aedat_file, os.path.join(origin_dataset_path, fname + '_labels.csv'), train_dir)

            for fname in trials_to_test_txt.readlines():
                fname = fname.strip()
                if fname.__len__() > 0:
                    aedat_file = os.path.join(origin_dataset_path, fname)
                    fname = os.path.splitext(fname)[0]  #remove .aedat
                    tpe.submit(split_aedat_files_to_npz, fname, aedat_file,
                                os.path.join(origin_dataset_path, fname + '_labels.csv'), test_dir)

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
    print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')


def create_data_set_events(npz_dir: str):
    train_data = []
    test_data = []
    train_label = []
    test_label = []

    npz_dir_list = os.listdir(npz_dir) #dir train,test
    for dir0 in npz_dir_list:
        dir1 = os.path.join(npz_dir,dir0) 
        classes_dir = os.listdir(dir1) #dir 0,1,2...
        for dir2 in classes_dir:
            file_dir = os.path.join(dir1,dir2) 
            filrs_names = os.listdir(file_dir)
            for file_name in filrs_names:
                npz_file = np.load(os.path.join(file_dir,file_name))
                t=npz_file['t']
                x=npz_file['x']
                y=npz_file['y']
                p=npz_file['p']
                txyp = {'t': t,'x': x,'y': y,'p': p}
                if dir0 =="train":
                    train_data.append(txyp)
                    train_label.append(int(dir2)) 
                else:
                    test_data.append(txyp)
                    test_label.append(int(dir2))
                    
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)


def create_same_directory_structure(source_dir: str, target_dir: str):
    for sub_dir_name in os.listdir(source_dir):
        source_sub_dir = os.path.join(source_dir, sub_dir_name)
        if os.path.isdir(source_sub_dir):
            target_sub_dir = os.path.join(target_dir, sub_dir_name)
            os.mkdir(target_sub_dir)
            print(f'Mkdir [{target_sub_dir}].')
            create_same_directory_structure(source_sub_dir, target_sub_dir)


def integrate_events_segment_to_frame(x: np.ndarray, y: np.ndarray, p: np.ndarray, H: int, W: int, j_l: int, j_r: int) -> np.ndarray:
    frame = np.zeros(shape=[2, H * W])
    x = x[j_l: j_r].astype(int)  # avoid overflow
    y = y[j_l: j_r].astype(int)
    p = p[j_l: j_r]
    mask = []
    mask.append(p == 0)
    mask.append(np.logical_not(mask[0]))
    for c in range(2):
        position = y[mask[c]] * W + x[mask[c]]
        events_number_per_pos = np.bincount(position)
        frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
    return frame.reshape((2, H, W))


def get_fixed_frames_number_segment_index(events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    N = events_t.size

    if split_by == 'number':
        di = N // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di
        j_r[-1] = N

    elif split_by == 'time':
        dt = (events_t[-1] - events_t[0]) // frames_num
        idx = np.arange(N)
        for i in range(frames_num):
            t_l = dt * i + events_t[0]
            t_r = t_l + dt
            mask = np.logical_and(events_t >= t_l, events_t < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1

        j_r[-1] = N
    else:
        raise NotImplementedError

    return j_l, j_r



def integrate_events_by_fixed_frames_number(events: Dict, label: str, split_by: str, frames_num: int, H: int, W: int) -> np.ndarray:
    '''
    :param events: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
    :param split_by: 'time' or 'number'
    :param frames_num: the number of frames
    :param H: the height of frame
    :param W: the weight of frame
    :return: frames,label
    Integrate events to frames by fixed frames number.
    '''
    t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
    j_l, j_r = get_fixed_frames_number_segment_index(t, split_by, frames_num)
    frames = np.zeros([frames_num, 2, H, W])
    for i in range(frames_num):
        frames[i] = integrate_events_segment_to_frame(x, y, p, H, W, j_l[i], j_r[i])
    return frames,label


def integrate_events_by_fixed_duration(events: Dict, duration: int, H: int, W: int) -> np.ndarray:
    '''
    Integrate events to frames by fixed time duration of each frame.
    '''
    x = events['x']
    y = events['y']
    t = events['t']
    p = events['p']
    N = t.size

    frames = []
    left = 0
    right = 0
    while True:
        t_l = t[left]
        while True:
            if right == N or t[right] - t_l > duration:
                break
            else:
                right += 1
        # integrate from index [left, right)
        frames.append(np.expand_dims(integrate_events_segment_to_frame(x, y, p, H, W, left, right), 0))

        left = right

        if right == N:
            return np.concatenate(frames)
        


def padding_zeros_with_variable_length(X):
    x_list = []
    x_len_list = []
    for x  in X:
        x_list.append(torch.as_tensor(x))
        x_len_list.append(x.shape[0])

    return torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True).numpy(), np.array(x_len_list)




def save_frame_to_gif(frames: torch.Tensor or np.ndarray, label:str, save_path: str,unique_ = "") -> None:
    '''
    frames: frames with `shape=[T, 2, H, W]`
         save frames to a gif file in the directory `save_path`
    '''
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    to_img = transforms.ToPILImage()
    img_tensor = torch.zeros([frames.shape[0], 3, frames.shape[2], frames.shape[3]])
    img_tensor[:, 1] = frames[:, 0]
    img_tensor[:, 2] = frames[:, 1]

    gif_frames = []
    for t in range(img_tensor.shape[0]):
        img = to_img(img_tensor[t])
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.title(f"Visualization frame_{t}")
        plt.savefig(os.path.join(save_path,f"frame_{t}.png"))
        plt.close()
        img_read = imageio.v2.imread(os.path.join(save_path,f"frame_{t}.png"))
        gif_frames.append(img_read)

    imageio.mimsave(os.path.join(save_path,f"visual_label_{label}"+f"{unique_}.gif"), # output gif
            gif_frames,          # array of input frames
            fps = 10)         # frames per second
    
    for file in os.listdir(save_path):  #remove png generated
        if(file[:5]=="frame"):
            os.remove(os.path.join(save_path,file))

    print(f'Save gif to [{save_path+f"/visual_label_{label}"+f"{unique_}.gif"}].')



def save_events_data_to_frame_to_dir(events_data: np.ndarray, events_data_label: np.ndarray,data_type:str, npz_save_path_root: str, split_by: str, frames_num: int, H: int, W: int, visual_save_path_root =None):
    num_data = len(events_data)
    for i in range(num_data):
        event_,label_ = events_data[i],events_data_label[i]
        frames_, label_ = integrate_events_by_fixed_frames_number(events=event_,label=label_,split_by=split_by,frames_num=frames_num,H=H,W=W)
        if visual_save_path_root is not None:
            save_frame_to_gif(frames_,label_,os.path.join(visual_save_path_root,data_type,str(label_)),unique_="_num_"+str(i))
        np.save(os.path.join(npz_save_path_root,data_type,str(label_),str(i)),frames_)
        print(f"save npy to [{os.path.join(npz_save_path_root,data_type,str(label_),str(i))}.npy]")


def save_events_data_to_frame_npz_by_fixed_frames_number(events_train_data: np.ndarray, events_train_label: np.ndarray, events_test_data: np.ndarray, events_test_label : np.ndarray, npz_save_path_root: str, split_by: str, frames_num: int, H: int, W: int, visual_save_path =None)-> None:
    assert frames_num > 0 and isinstance(frames_num, int)
    assert split_by == 'time' or split_by == 'number' or split_by == "fixed_duration"
    if split_by=="number":
        npz_root_path = os.path.join(npz_save_path_root,"split_by_number")
    elif split_by=="time":
        npz_root_path = os.path.join(npz_save_path_root,"split_by_time")

    if visual_save_path is not None:
        if split_by=="number":
            visual_save_path = os.path.join(visual_save_path,"split_by_number")
        elif split_by=="time":
            visual_save_path = os.path.join(visual_save_path,"split_by_time")

    t_ckp = time.time()
    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 16)) as tpe:
        print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
        tpe.submit(save_events_data_to_frame_to_dir,events_train_data,events_train_label,"train",npz_root_path,split_by,frames_num,H,W,visual_save_path)
        tpe.submit(save_events_data_to_frame_to_dir,events_test_data,events_test_label,"test",npz_root_path,split_by,frames_num,H,W,visual_save_path)
    #save_events_data_to_frame_to_dir(events_train_data,events_train_label,"train",npz_root_path,split_by,frames_num,H,W,visual_save_path)
    #save_events_data_to_frame_to_dir(events_test_data,events_test_label,"test",npz_root_path,split_by,frames_num,H,W,visual_save_path)

    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
    print(f'All files have been saved.')



def save_events_data_to_frame_npy_by_fixed_time_duration(events_train_data: np.ndarray, events_train_label: np.ndarray, events_test_data: np.ndarray, events_test_label : np.ndarray, npy_save_path_root: str,durantion:int, H: int, W: int, visual_save_path =None):
    len_train = len(events_train_data)
    len_test = len(events_test_data)
    data = np.concatenate((events_train_data,events_test_data))
    data_label = np.concatenate((events_train_label,events_test_label))
    train_save_path = os.path.join(npy_save_path_root,"split_by_fixed_duration","train")
    test_save_path = os.path.join(npy_save_path_root,"split_by_fixed_duration","test")
    frames_variable_len_list = []

    for e in data:
        frames_variable_len_list.append(integrate_events_by_fixed_duration(e,durantion,H,W)) 
    frames_padding,frames_origin_len_list = padding_zeros_with_variable_length(frames_variable_len_list)
    t_ckp = time.time()

    for i in range(len_train):
        np.save(os.path.join(train_save_path,str(data_label[i]),str(i)),frames_padding[i])
        print(f"save npy to [{os.path.join(train_save_path,str(data_label[i]),str(i))}.npy]") 
        if visual_save_path is not None:
            save_frame_to_gif(frames_padding[i],data_label[i],os.path.join(visual_save_path,"split_by_fixed_duration","train",str(data_label[i])),unique_="_num_"+str(i))
    for i in range(len_train,len_train+len_test):
        np.save(os.path.join(test_save_path,str(data_label[i]),str(i)),frames_padding[i])
        print(f"save npy to [{os.path.join(test_save_path,str(data_label[i]),str(i))}.npy]")
        if visual_save_path is not None:
            save_frame_to_gif(frames_padding[i],data_label[i],os.path.join(visual_save_path,"split_by_fixed_duration","test",str(data_label[i])),unique_="_num_"+str(i))

    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
    print(f'All files have been saved.')
    

def create_data_set_frames(npy_dir: str):
    train_data = []
    test_data = []
    train_label = []
    test_label = []

    npz_dir_list = os.listdir(npy_dir) #dir train,test
    for dir0 in npz_dir_list:
        dir1 = os.path.join(npy_dir,dir0) 
        classes_dir = os.listdir(dir1) #dir 0,1,2...
        for dir2 in classes_dir:
            file_dir = os.path.join(dir1,dir2) 
            filrs_names = os.listdir(file_dir)
            for file_name in filrs_names:
                npy = np.load(os.path.join(file_dir,file_name))
                if dir0 =="train":
                    train_data.append(npy)
                    train_label.append(int(dir2)) 
                else:
                    test_data.append(npy)
                    test_label.append(int(dir2))
                    
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)



classify_aedat_to_npz(dvs_path,events_dvs_split_path)

train_data,train_label,test_data,test_label = create_data_set_events(events_dvs_split_path)

os.mkdir(os.path.join(frames_dvs_split_path,"split_by_number"))
os.mkdir(os.path.join(frames_dvs_split_path,"split_by_time"))
os.mkdir(os.path.join(frames_dvs_split_path,"split_by_fixed_duration"))

create_same_directory_structure(events_dvs_split_path, os.path.join(frames_dvs_split_path,"split_by_number"))
create_same_directory_structure(events_dvs_split_path, os.path.join(frames_dvs_split_path,"split_by_time"))
create_same_directory_structure(events_dvs_split_path, os.path.join(frames_dvs_split_path,"split_by_fixed_duration"))


save_events_data_to_frame_npz_by_fixed_frames_number(train_data,train_label,test_data,test_label,frames_dvs_split_path,"number",20,128,128)
save_events_data_to_frame_npz_by_fixed_frames_number(train_data,train_label,test_data,test_label,frames_dvs_split_path,"time",20,128,128)
save_events_data_to_frame_npy_by_fixed_time_duration(train_data,train_label,test_data,test_label,frames_dvs_split_path,500000,128,128)




train_data_frames, train_label_frames, test_data_frames, test_label_frames = create_data_set_frames(os.path.join(frames_dvs_split_path,"split_by_fixed_duration"))
print("Start saving frames npy files split_by_fixed_duration")
save_path=os.path.join(frames_dvs_split_path,"split_by_fixed_duration")
np.save(os.path.join(save_path,"train_data_frames_fixed_duration"),np.float32(train_data_frames))
np.save(os.path.join(save_path,"train_label_frames_fixed_duration"),np.float32(train_label_frames))
np.save(os.path.join(save_path,"test_data_frames_fixed_duration"),np.float32(test_data_frames))
np.save(os.path.join(save_path,"test_label_frames_fixed_duration"),np.float32(test_label_frames))

train_data_frames, train_label_frames, test_data_frames, test_label_frames = create_data_set_frames(os.path.join(frames_dvs_split_path,"split_by_number"))
print("Start saving frames npy files split_by_number")
save_path=os.path.join(frames_dvs_split_path,"split_by_number")
np.save(os.path.join(save_path,"train_data_frames_number"),np.float32(train_data_frames))
np.save(os.path.join(save_path,"train_label_frames_number"),np.float32(train_label_frames))
np.save(os.path.join(save_path,"test_data_frames_number"),np.float32(test_data_frames))
np.save(os.path.join(save_path,"test_label_frames_number"),np.float32(test_label_frames))

train_data_frames, train_label_frames, test_data_frames, test_label_frames = create_data_set_frames(os.path.join(frames_dvs_split_path,"split_by_time"))
print("Start saving frames npy files split_by_time")
save_path=os.path.join(frames_dvs_split_path,"split_by_time")
np.save(os.path.join(save_path,"train_data_frames_time"),np.float32(train_data_frames))
np.save(os.path.join(save_path,"train_label_frames_time"),np.float32(train_label_frames))
np.save(os.path.join(save_path,"test_data_frames_time"),np.float32(test_data_frames))
np.save(os.path.join(save_path,"test_label_frames_time"),np.float32(test_label_frames))
