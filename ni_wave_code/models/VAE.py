from tensorflow.keras import layers, Model, Input, losses
from tensorflow.keras.layers import concatenate as concat
import tensorflow as tf
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from scipy.stats import pearsonr




# ___ SAMPLING LAYER
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ___ ENCODER LAYER
class Encoder(layers.Layer):
    def __init__(self, latent_dim: object = 4, inter1_dim: object = 8, inter2_dim: object = 8, act: object = 'relu', name: object = 'encoder', **kwargs: object) -> object:
        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder_1 = layers.Dense(inter1_dim, activation=act)
        self.encoder_2 = layers.Dense(inter2_dim, activation=act)
        self.encoder_mean = layers.Dense(latent_dim)
        self.encoder_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x1 = self.encoder_1(inputs)
        x2 = self.encoder_2(x1)
        z_mean = self.encoder_mean(x2)
        z_log_var = self.encoder_log_var(x2)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


# ___ DECODER LAYER
class Decoder(layers.Layer):
    def __init__(self, original_dim, inter1_dim=8, inter2_dim=8, act='relu', name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder_2 = layers.Dense(inter2_dim, activation=act)
        self.decoder_1 = layers.Dense(inter1_dim, activation=act)
        #self.decoded = layers.Dense(original_dim, activation='tanh')
        self.decoded = layers.Dense(original_dim)

    def call(self, inputs):
        x2 = self.decoder_2(inputs)
        x1 = self.decoder_1(x2)
        return self.decoded(x1)


# ___ VAE FUNCTION
def BetaVAutoEncoder(original_dim, latent_dim=4, inter1_dim=8, inter2_dim=8, act='relu', beta=1):
    '''
    :param original_dim: input dimension
    :param latent_dim: dimension of (latent) representation layer (default = 4)
    :param inter1_dim: first layer in encoder (symmetric decoder) (default = 8)
    :param inter2_dim: second layer in encoder (symmetric decoder) (default = 8)
    :param act: activation function (default = 'relu')
    :return: tf.keras.Model(s) encoder, decoder, vae
    '''
    original_inputs = Input(shape=(original_dim,), name='encoder_input')

    # encoder model
    z_mean, z_log_var, z = Encoder(latent_dim=latent_dim,
                                   inter1_dim=inter1_dim,
                                   inter2_dim=inter2_dim,
                                   act=act)(original_inputs)
    encoder = Model(inputs=original_inputs, outputs=[z_mean, z_log_var, z], name='encoder')

    # decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

    reconstruction = Decoder(original_dim,
                             inter1_dim=inter1_dim,
                             inter2_dim=inter2_dim,
                             act=act)(latent_inputs)
    decoder = Model(inputs=latent_inputs, outputs=reconstruction, name='decoder')

    # define VAE
    outputs = decoder(z)
    vae = Model(inputs=original_inputs, outputs=outputs, name='vae')
    # add KL loss
    kl_loss = -0.5 * (z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    kl_metric = tf.reduce_sum(kl_loss, axis=1)
    kl_loss = 1/original_dim * beta * tf.reduce_mean(kl_metric)

    vae.add_loss(kl_loss)
    vae.add_metric(kl_metric, name='kl_loss', aggregation='mean')
    return encoder, decoder, vae

########################################################################################################
########################################################################################################

# TFP
import tensorflow_probability as tfp
tfd = tfp.distributions

class VAE_TFP:

    def __init__(self, latent_dim, beta, learning_rate, original_dim, inter1_dim, inter2_dim=0, act='relu'):
        self.dim_x = original_dim
        self.dim_z = latent_dim
        self.kl_weight = beta
        self.learning_rate = learning_rate
        self.inter1_dim = inter1_dim
        self.inter2_dim = inter2_dim
        self.act = act

    # Sequential API encoder
    def encoder_z(self):
        # define prior distribution for the code, which is an isotropic Gaussian
        '''
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.dim_z), scale=1.),
                                reinterpreted_batch_ndims=1)
        '''
        prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.dim_z))
        # build layers argument for tfk.Sequential()
        input_shape = self.dim_x
        layers = [tf.keras.layers.InputLayer(input_shape=(input_shape,))]
        layers.append(tf.keras.layers.Dense(self.inter1_dim, activation=self.act))
        if self.inter2_dim > 0:
            layers.append(tf.keras.layers.Dense(self.inter2_dim, activation=self.act))
        # the following two lines set the output to be a probabilistic distribution
        layers.append(tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(self.dim_z),
                                 activation=None, name='z_params'))
        layers.append(tfp.layers.IndependentNormal(self.dim_z,
                                                   convert_to_tensor_fn=tfd.Distribution.sample,
                                                   activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=self.kl_weight),
                                                   name='z_layer'))
        return tf.keras.Sequential(layers, name='encoder')


    # Sequential API decoder
    def decoder_x(self):
        layers = [tf.keras.layers.InputLayer(input_shape=self.dim_z)]
        if self.inter2_dim > 0:
            layers.append(tf.keras.layers.Dense(self.inter2_dim, activation=self.act))
        layers.append(tf.keras.layers.Dense(self.inter1_dim, activation=self.act))
        layers.append(tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(self.dim_x),
                                 activation=None, name='x_params'))
        layers.append(tfp.layers.IndependentNormal(self.dim_x, name='x_layer'))
        return tf.keras.Sequential(layers, name='decoder')

    def build_vae_keras_model(self):
        x_input = tf.keras.Input(shape=(self.dim_x,))
        encoder = self.encoder_z()
        decoder = self.decoder_x()
        z = encoder(x_input)

        # compile VAE model
        model = tf.keras.Model(inputs=x_input, outputs=decoder(z))
        model.compile(loss=negative_log_likelihood,
                      optimizer=tf.keras.optimizers.Adam(self.learning_rate))
        return model, encoder, decoder

# the negative of log-likelihood for probabilistic output
negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

def mean_loglik(model, data):
    out = model(data)
    assert isinstance(out, tfd.Distribution)
    loglik = out.log_prob(data)
    loglik_mean = np.asarray(loglik).mean()
    return loglik_mean

def mean_KL(encoder, data, latent_dim):
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1.),
                            reinterpreted_batch_ndims=1)
    z = encoder(data)
    assert isinstance(z, tfd.Distribution)
    KL = tfp.distributions.kl_divergence(prior, z)
    KL_mean = np.asarray(KL).mean()
    return KL_mean

def learn(df, inter1_dim, inter2_dim, latent_dim, beta, learning_rate, list, callback, pretrain_epochs, batch_size):
    mod = '/model_{}.{}.{}_{}_{}'.format(inter1_dim, inter2_dim, latent_dim, beta, learning_rate)
    vae_class = VAE_TFP(latent_dim=latent_dim,
                        beta=beta,
                        learning_rate=learning_rate,
                        original_dim=df.shape[1],
                        inter1_dim=inter1_dim,
                        inter2_dim=inter2_dim,
                        act='relu')
    vae, en, de = vae_class.build_vae_keras_model()
    vae.fit(df, df, batch_size=batch_size, epochs=pretrain_epochs, callbacks=[callback])

    elbo = vae.evaluate(df,df)
    loglik = -mean_loglik(vae, df)
    KL = mean_KL(en, df, latent_dim)

    res = {'name': mod, 'inter1_dim': inter1_dim, 'inter2_dim': inter2_dim,
           'latent_dim': latent_dim, 'beta': beta, 'learning_rate': learning_rate,
           'elbo': elbo, 'mean_loglik': loglik, 'mean_KL': KL}
    list.append(res)
    return vae, en, de, mod

def load_vae(path, learning_rate):
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    vae = tf.keras.models.load_model(path+'/vae.h5',
                                     custom_objects={negative_log_likelihood:negative_log_likelihood},
                                     compile=False)
    vae.compile(loss=negative_log_likelihood,
                optimizer=tf.keras.optimizers.Adam(learning_rate))
    vae.load_weights(path+'/weights.h5')
    encoder = tf.keras.models.load_model(path+'/en.h5')
    decoder = tf.keras.models.load_model(path+'/de.h5')

    return vae, encoder, decoder



########################################################################################################
########################################################################################################

# ___ CVAE FUNCTION
def CVAE(original_dim, latent_dim=4, inter1_dim=8, inter2_dim=8, act='relu', beta=1):
    '''
    :param original_dim: input dimension
    :param latent_dim: dimension of (latent) representation layer (default = 4)
    :param inter1_dim: first layer in encoder (symmetric decoder) (default = 8)
    :param inter2_dim: second layer in encoder (symmetric decoder) (default = 8)
    :param act: activation function (default = 'relu')
    :return: tf.keras.Model(s) encoder, decoder, vae
    '''
    # inputs
    original_input = Input(shape=(original_dim, ), name='original_input')
    label_input = Input(shape=(1,), name='label_input')
    input = concat([original_input, label_input])
    # encoder model
    z_mean, z_log_var, z = Encoder(latent_dim=latent_dim,
                                   inter1_dim=inter1_dim,
                                   inter2_dim=inter2_dim,
                                   act=act)(input)
    encoder = Model(inputs=input, outputs=[z_mean, z_log_var, z], name='encoder')

    # decoder model
    latent_input = Input(shape=(latent_dim+1, ), name='de_input')
    zc_input = concat([z, label_input])
    reconstruction = Decoder(original_dim+1,
                             inter1_dim=inter1_dim,
                             inter2_dim=inter2_dim,
                             act=act)(latent_input)
    decoder = Model(inputs=latent_input, outputs=reconstruction, name='decoder')


    # define VAE
    outputs = decoder(zc_input)
    vae = Model(inputs=[original_input, label_input], outputs=outputs, name='vae')
    # add KL loss
    kl_loss = 1/original_dim * -0.5 * (z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    kl_loss = beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

    #kl_loss = 1/original_dim * beta * -0.5 * tf.reduce_mean(
    #    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
    #)
    vae.add_loss(kl_loss)
    vae.add_metric(kl_loss, name='kl_loss')
    return encoder, decoder, vae

# ___ sample function
def sampleVAE(de_model, en_prediction, num_samples, label, scale, both=False, conditional=False, seed=43):
    np.random.seed(seed)
    if both and not conditional:
        print("Both is only possible for conditional VAE")
        return

    z = en_prediction
    Sigma = np.cov(np.matrix.transpose(z))
    Mean = np.mean(z, axis=0)
    samples = np.random.multivariate_normal(Mean, Sigma, size=num_samples, check_valid='warn', tol=1e-8)

    if conditional:
        if both:
            sample_other = np.abs(label-1)
            samples_label = np.empty([num_samples, 1])
            for i in range(num_samples):
                samples_label[i] = sample_other
            samples_other = np.concatenate([samples, samples_label], axis=1)

        sample_labels = label
        samples_label = np.empty([num_samples, 1])
        for i in range(num_samples):
            samples_label[i] = sample_labels
        samples = np.concatenate([samples, samples_label], axis=1)

    gen_out_st = de_model.predict(samples)

    if conditional:
        gen_out_st = gen_out_st[:, :-1]
        if both:
            gen_out_st_other = de_model.predict(samples_other)
            gen_out_st_other = gen_out_st_other[:, :-1]

    gen_out = pd.DataFrame(scale.inverse_transform(gen_out_st))
    if both:
        gen_out_other = pd.DataFrame(scale.inverse_transform(gen_out_st_other))
        gen_out = gen_out.append(gen_out_other)

    return gen_out

def sampleVAE1D(de_model, en_prediction, scale, dim, num_samples_other=1, seed=43, tfp=True):
    np.random.seed(seed)
    z = np.asarray(en_prediction)
    Sigma = np.cov(np.matrix.transpose(z))
    Mean = np.mean(z, axis=0)
    sample = np.random.multivariate_normal(Mean, Sigma, size=num_samples_other,
                                           check_valid='warn', tol=1e-8).reshape(num_samples_other, en_prediction.shape[1])
    dim_values = np.arange(start=z[dim].min(),
                            stop=z[dim].max(),
                            step=(z[dim].max()-z[dim].min())/50)

    gen_out = []
    for i in range(50):
        sample_temp = sample
        for j in range(num_samples_other):
            sample_temp[j][dim] = dim_values[i]
        if tfp:
            pred_temp_dist = de_model(sample_temp)
            assert isinstance(pred_temp_dist, tfd.Distribution)
            pred_temp = np.asarray(pred_temp_dist.mean())
        else:
            pred_temp = de_model.predict(sample_temp)
        scaled_temp = scale.inverse_transform(pred_temp)
        out_temp = np.append(sample_temp, pred_temp, axis=1)
        out_temp = np.append(out_temp, scaled_temp, axis=1)

        gen_out.append(out_temp)
    gen_out = np.array(gen_out)
    gen_out = gen_out.reshape(-1, sample.shape[-1]+2*pred_temp.shape[-1])

    col_names = []
    for i in range(en_prediction.shape[-1]):
        col_names.append('sample_dim{}'.format(i))
    for i in range(pred_temp.shape[-1]):
        col_names.append('pred_or{}'.format(i))
    for i in range(pred_temp.shape[-1]):
        col_names.append('pred_scale{}'.format(i))
    gen_out = pd.DataFrame(gen_out, columns=col_names)
    gen_out['dim'] = dim

    return gen_out

def sample_slopes(sample_ld, sample_dim, latent_dim, original_dim, names):
    out = []
    data = np.asarray(sample_ld)
    latent = data[:, sample_dim].reshape(sample_ld.shape[0], 1)
    col_names = ['variable', 'm_dim{}'.format(sample_dim),'p_dim{}'.format(sample_dim)]
    for v in range(original_dim):
        #var = data[:, (latent_dim+v)].reshape(sample_ld.shape[0], 1)
        lr = ols('pred_or{} ~ sample_dim{}'.format(v, sample_dim), data=sample_ld).fit()
        m = round(lr.params[1], 3)
        p = round(lr.pvalues[1],4)
        variable = sample_ld.columns[latent_dim+v]
        out.append([variable, m,p])
    out_pd = pd.DataFrame(out, columns=col_names, index=names)
    return out_pd

def sample_corr(sample_ld, sample_dim, latent_dim, original_dim):
    out = []
    for i in range(original_dim):
        temp = np.asarray(pearsonr(sample_ld.iloc[:,sample_dim],sample_ld.iloc[:,(latent_dim+i)]))
        out.append(temp)
    output = pd.DataFrame(out, columns=['r_dim{}'.format(sample_dim), 'p_dim{}'.format(sample_dim)])
    return output