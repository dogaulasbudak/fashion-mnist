import tensorflow as tf
import tensorflow_datasets as tfds


(ds_train, ds_test), ds_info = tfds.load('fashion_mnist', split=['train', 'test'], shuffle_files=True
                                         , as_supervised=True, with_info=True)


def normalize_images(image, label):
    """Normalizes the images"""
    return tf.cast(image, tf.float32)/255., label

ds_train = ds_train.map(normalize_images)
ds_train = ds_train.shuffle(60000, seed=42)
ds_train = ds_train.batch(1)


ds_test = ds_test.map(normalize_images)
ds_test = ds_test.shuffle(10000, seed=42)
ds_test = ds_test.batch(1)


#The model 

class CNN_classifier(tf.keras.Model):
    def __init__(self):
        super(CNN_classifier, self).__init__()
        self.norm = tf.keras.layers.Normalization()
        self.conv1 = tf.keras.layers.Conv2D(2, 2, input_shape=(28, 28, 1), padding='same', activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D((2,2))
        self.conv2 = tf.keras.layers.Conv2D(4, 2, padding='same', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D((2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)
        
    def call(self, inputs, training=True):
        x = self.norm(inputs)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x


loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

model = CNN_classifier()

train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()




@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predic = model(images, training=True)
        loss = loss_function(labels, predic)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predic)
    
@tf.function
def test_step(images, labels):
    predic = model(images, training=False)
    loss = loss_function(labels, predic)
    
    test_loss(loss)
    test_accuracy(labels, predic)


epochs = 5

for epoch in range(epochs):
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()
    
    for images, labels in ds_train:
        train_step(images, labels)
        
    for images, labels in ds_test:
        test_step(images, labels)
        
    print(f'Epoch {epoch + 1}, '
          f'Loss: {train_loss.result()}, '
          f'Accuracy: {train_accuracy.result() * 100}, '
          f'Test Loss: {test_loss.result()}, '
          f'Test Accuracy: {test_accuracy.result() * 100}')