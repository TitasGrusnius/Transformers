import tensorflow as tf

#@tf.function
def train_one_epoch(model, criterion, optimizers, images, boxes, labels):

    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        loss_dict = criterion(outputs, boxes, labels)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        # loss_value = losses_reduced_scaled.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     print(loss_dict_reduced)
        #     sys.exit(1)

        # total_loss = total_loss / gradient_aggregate

    # # Compute gradient for each part of the network
    # gradient_steps = gather_gradient(model, optimizers, total_loss, tape, config, log)

    return losses