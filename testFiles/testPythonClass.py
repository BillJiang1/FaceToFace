import tensorflow as tf


class TestClass:
    # 结果表明A,B不是同一个对象
    def __init__(self):
        A = self.A()
        B = self.A()
        print(A)
        print(B)

    def A(self):
        input_ = tf.keras.Input(shape=(8, 8, 512))
        x = input_
        x = tf.keras.layers.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return tf.keras.Model(input_, x)


if __name__ == '__main__':
    test = TestClass()
