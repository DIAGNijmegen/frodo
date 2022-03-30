import tensorflow as tf
import tensorflow_probability as tfp

from tf_frodo.utils import create_feature_extractor


class FRODO:
    def __init__(self, model, dtype=tf.float32):
        super().__init__()
        self.feature_extractor = create_feature_extractor(model, retain_outputs=True)
        self.extracted_features = None
        self.means, self.stds, self.inverse_covariances = None, None, None
        self.dtype = dtype

    def fit(self, data):
        extracted_feature_list = self.feature_extractor(data)["layer_outputs"]
        extracted_feature_list = self.cast_all(extracted_feature_list)
        means = [tf.reduce_mean(features, axis=0, keepdims=False) for features in extracted_feature_list]
        stds = [tf.math.reduce_std(features, axis=0) for features in extracted_feature_list]

        zero_stds = [std == 0 for std in stds]
        zero_stds = self.cast_all(zero_stds)

        stds = [std + zero_std * tf.ones_like(std, dtype=self.dtype) for std, zero_std in zip(stds, zero_stds)]

        normalized_features = [(features - mean) / std for mean, std, features in
                               zip(means, stds, extracted_feature_list)]

        covariances = [tfp.stats.covariance(features) for features in normalized_features]
        Is = [tf.eye(mean.shape[-1], dtype=self.dtype) for mean in means]
        epsilon = tf.constant(0.01, dtype=self.dtype)
        covariances = [covariance + (I * epsilon) for covariance, I in zip(covariances, Is)]
        inverse_covariances = [tf.linalg.inv(covariance, adjoint=False, name=None) for covariance in covariances]

        self.means = [tf.identity(mean) for mean in means]
        self.stds = [tf.identity(std) for std in stds]
        self.inverse_covariances = [tf.identity(inverse_covariance) for inverse_covariance in inverse_covariances]
        return self.apply_frodo_to_feature_extractor()

    def apply_frodo_to_feature_extractor(self):
        num_frodo_outputs = len(self.means)
        layer_outputs = self.feature_extractor.outputs[:num_frodo_outputs]
        model_outputs = self.feature_extractor.outputs[num_frodo_outputs:]
        normalized_outputs = [((output - mean) / std)
                              for mean, std, output in zip(self.means, self.stds, layer_outputs)]
        num_features = [output.shape[-1]
                        for output in normalized_outputs]
        normalized_outputs = [tf.expand_dims(normalized_output, -1) for normalized_output in normalized_outputs]
        mahalanobis_distances = [
            tf.matmul(normalized_output, tf.matmul(inverse_covariance, normalized_output), transpose_a=True)
            for normalized_output, inverse_covariance in zip(normalized_outputs, self.inverse_covariances)]
        mahalanobis_distances = [tf.squeeze(md) for md in mahalanobis_distances]
        mahalanobis_distances = [tf.sqrt(md) for md in mahalanobis_distances]

        globally_normalized_mahalanobis_distances = [md / num_feature
                                                     for md, num_feature in zip(mahalanobis_distances, num_features)]

        frodo = tf.reduce_mean(globally_normalized_mahalanobis_distances, axis=0)
        return tf.keras.models.Model(inputs=self.feature_extractor.inputs,
                                     outputs={"model_output": model_outputs, "FRODO": frodo})

    def cast_all(self, list_of_tensors):
        return [tf.cast(tensor, dtype=self.dtype) for tensor in list_of_tensors]
