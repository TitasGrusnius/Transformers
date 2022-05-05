import tensorflow as tf

@tf.function
def run_train_step(model, images, boxes, labels):

    with tf.GradientTape() as tape:
        m_outputs = model(images, training=True)
        # total_loss, log = get_losses(m_outputs, t_bbox, t_class, config)
        # total_loss = total_loss / gradient_aggregate

    # # Compute gradient for each part of the network
    # gradient_steps = gather_gradient(model, optimizers, total_loss, tape, config, log)

    return m_outputs