# class Test_class(object):
#     def __init__(self):
#         names = self.__dict__
#         for i in range(5):
#             names['n' + str(i)] = i


# t = Test_class()
# print(t.n0, t.n1, t.n2, t.n3, t.n4)
for i in range(5):
    exec('var{} = {}'.format(i, i))
exec('var{} = {}'.format(i, i))
print(var0, var1, var2, var3, var4)
