import math
import sys

import tensorflow as tf
import utils.misc as utils

def train_one_epoch(model, criterion, optimizers, images, boxes, labels):

    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        loss_dict = criterion(outputs, boxes, labels)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.numpy()
        loss_reduced = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(loss_value, axis=2), axis=1))
       
        if not math.isfinite(loss_reduced):
            print("Loss is {}, stopping training".format(loss_reduced))
            print(loss_dict_reduced)
            sys.exit(1)

        trainable_variables = []
        backbone_trainable_variables = model.backbone.layers[0].get_trainable_variables()
        transformer_trainable_variables = model.transformer.trainable_weights
        fnn_trainable_variables = model.get_trainable_weights()
        trainable_variables = backbone_trainable_variables  + transformer_trainable_variables + fnn_trainable_variables
    
    gradients = tape.gradient(loss_reduced, fnn_trainable_variables)

    print(gradients)
    # fnn_gradient = gradients[len(backbone_trainable_variables)+len(transformer_trainable_variables):]
    # backbone_gradient = gradients[:len(backbone_trainable_variables)]
    # transformer_gradient = gradients[len(backbone_trainable_variables):len(backbone_trainable_variables)+len(transformer_trainable_variables)]
  
    # Update weights
    optimizers['fnn'].apply_gradients(zip(gradients, fnn_trainable_variables))
    # optimizers['transformer'].apply_gradients(zip(transformer_gradient, transformer_trainable_variables))
    # optimizers['backbone'].apply_gradients(zip(backbone_gradient, backbone_trainable_variables))



# @tf.stop_gradient()
# def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
#     model.eval()
#     criterion.eval()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Test:'

#     iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
#     coco_evaluator = CocoEvaluator(base_ds, iou_types)
#     # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

#     panoptic_evaluator = None
#     if 'panoptic' in postprocessors.keys():
#         panoptic_evaluator = PanopticEvaluator(
#             data_loader.dataset.ann_file,
#             data_loader.dataset.ann_folder,
#             output_dir=os.path.join(output_dir, "panoptic_eval"),
#         )

#     for samples, targets in metric_logger.log_every(data_loader, 10, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         outputs = model(samples)
#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])

#         orig_target_sizes = tf.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes)
#         if 'segm' in postprocessors.keys():
#             target_sizes = tf.stack([t["size"] for t in targets], dim=0)
#             results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)

#         if panoptic_evaluator is not None:
#             res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#             for i, target in enumerate(targets):
#                 image_id = target["image_id"].item()
#                 file_name = f"{image_id:012d}.png"
#                 res_pano[i]["image_id"] = image_id
#                 res_pano[i]["file_name"] = file_name

#             panoptic_evaluator.update(res_pano)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#     if panoptic_evaluator is not None:
#         panoptic_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#     panoptic_res = None
#     if panoptic_evaluator is not None:
#         panoptic_res = panoptic_evaluator.summarize()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     if coco_evaluator is not None:
#         if 'bbox' in postprocessors.keys():
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in postprocessors.keys():
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#     if panoptic_res is not None:
#         stats['PQ_all'] = panoptic_res["All"]
#         stats['PQ_th'] = panoptic_res["Things"]
#         stats['PQ_st'] = panoptic_res["Stuff"]
#     return stats, coco_evaluator
