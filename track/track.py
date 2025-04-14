import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load1
from yolov7.utils.datasets import LoadStreams
from yolov7.utils.general import (check_img_size, scale_coords,
                                  increment_path, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

import cv2
import torch
from ultralytics import YOLO

@torch.no_grad()
def run(
        source='sources.txt',
        yolo_weights=WEIGHTS / 'yolo11n.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'resnet50_market1501_aicity156.onnx',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        base_path=ROOT / 'preds/',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)

    # save_dir = increment_path(Path(project) / 'exp', exist_ok=exist_ok)  # increment run
    # save_dir = Path(save_dir)
    # (save_dir / 'tracks').mkdir(parents=True, exist_ok=True)  # make dir

    # save_dir = increment_path(Path(project) / 'exp', exist_ok=exist_ok)  # increment run
    save_dir = Path(base_path)
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)

    # WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = YOLO(yolo_weights)

    names, = model.names,

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=32)
    nr_sources = len(dataset.sources)
 
    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources 
    print("STRONGSORT WEIGHTS: ", strong_sort_weights)
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        # strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    
    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        start_time = time.time()
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, imgsz=640, conf=0.7, classes=0)
        t3 = time_synchronized()
        dt[1] += t3 - t2 
        # Apply NMS
        dt[2] += time_synchronized() - t3
        # Process detections
        for i, (det) in enumerate(pred):  # detections per image
            det = det.boxes
            seen += 1
            # if webcam:  # nr_sources >= 1
            p, im0, _ = path[i], im0s[i].copy(), dataset.count
            p = Path(p)  # to Path
            s += f'{i}: '
            # txt_file_name = p.name
            txt_file_name = str(frame_idx)
            save_path = save_dir / p.stem
            save_path.mkdir(parents=True, exist_ok=True)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_path / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det_xyxy = det.xyxy
                det_xyxy = det_xyxy.clone()
                det_xyxy = scale_coords(im.shape[2:], det_xyxy, im0.shape).round()

                confs = det.conf
                clss = det.cls

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(det_xyxy, confs.cpu(), clss.cpu(), im0)
                # outputs[i] = strongsort_list[i].update(xywhs, confs, clss, im0)
                t5 = time_synchronized()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:

                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        emb = output[7]
                        emb = list((emb[0]))

                        write_txt = "{"+f"'sensorId':'{p.name}', 'frameId':'{frame_idx}', 'personId':'{id}', 'bbox':'[{output[0]}, {output[1]}, {output[2]}, {output[3]}]', 'confidence':'{conf}', 'embedding':'{emb}'"+"}\n"
                        # write_txt = "{"+f"'sensorId':'{p.name}', 'frameId':'{frame_idx}', 'personId':'{id}', 'bbox':'[{output[0]}, {output[1]}, {output[2]}, {output[3]}]', 'confidence':'{conf}'"+"}\n"
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(write_txt)

                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                with open(txt_path + '.txt', 'a') as f:
                    f.write("{}")
                print('No detections')


            prev_frames[i] = curr_frames[i]
        print("Frame Id: ",frame_idx, "Time taken: ", time.time()- start_time)
        start_time = time.time()
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
    if save_vid:
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    # parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'resnet50_market1501_aicity156.onnx')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='../sources.txt', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--base-path', default=ROOT / 'preds', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
