class A():
    def __init__(self):
        self.a = 100

class B(A):
    def __init__(self):
        A.__init__(self)
        # super(A, self).__init__()


b = B()
print(b.a)

