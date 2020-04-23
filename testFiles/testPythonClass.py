import tensorflow as tf


class TestClass:

    def AA(self,a,b,c=0):
        print(a,b,c)


if __name__ == '__main__':
    test = TestClass()
    test.AA(1,2)
