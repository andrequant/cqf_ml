import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

mean = data['returns'].mean()
std = data['returns'].std()
print(mean, std)


fcb = [tf.contrib.layers.bucketized_column(fc,
                    boundaries=[-0.0005, 0.0001, 0.0005])]
                    
model = tf.contrib.learn.DNNClassifier(hidden_units=[50, 50],
                                       feature_columns=fcb)
                                       
def get_data():
  fc = {'returns': tf.constant(data[cols].values)}
  la = tf.constant((data['returns'] > 0).astype(int).values,
                   shape=[len(data), 1])
  return fc, la
    
    
model.fit(input_fn=get_data, steps=100)
   
model.evaluate(input_fn=get_data, steps=1)
   
   
data['dnn_pred'] = list(model.predict(input_fn=get_data))
data['dnn_pred'] = np.where(data['dnn_pred'] > 0, 1.0, -1.0)

data['dnn_returns'] = data['returns'] * data['dnn_pred']

data[['returns', 'ols_returns', 'log_returns', 'dnn_returns']].cumsum(
        ).apply(np.exp).plot(figsize=(10, 6));
