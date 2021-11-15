class LearnClass:
    static_element = 123

    def __init__(self):
        self.init_element = 456

c1 = LearnClass()

print('c1 = LearnClass')
print('c1.static_element: ', c1.static_element)
print('c1.init_element: ', c1.init_element)

c3 = LearnClass()

c3.static_element = 888

LearnClass.static_element = 999

print('LearnClass.static_element = 999')
print('c1 = LearnClass')
print('c1.static_element: ', c1.static_element)
print('c1.init_element: ', c1.init_element)

c2 = LearnClass()
print('c2 = LearnClass')
print('c2.static_element: ', c2.static_element)
print('c2.init_element: ', c2.init_element)

print('c3 = LearnClass')
print('c3.static_element: ', c3.static_element)
print('c3.init_element: ', c3.init_element)