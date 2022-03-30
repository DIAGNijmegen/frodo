import tensorflow as tf


def create_feature_extractor(model, retain_outputs=False):
    layer_outputs = []
    sub_models = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model):
            sub_models.append(create_feature_extractor(layer))
        if isinstance(layer, tf.keras.layers.Conv2D):
            averaged_layer_outputs = tf.reduce_mean(layer.output, axis=[1, 2])
            layer_outputs.append(averaged_layer_outputs)

    for sub_model in sub_models:
        layer_outputs.extend(sub_model(model.inputs))

    model_outputs = None

    if retain_outputs:
        model_outputs = {"layer_outputs": layer_outputs, "model_outputs": model.outputs}
    else:
        model_outputs = layer_outputs

    return tf.keras.models.Model(inputs=model.inputs, outputs=model_outputs)
