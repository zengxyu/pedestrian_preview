class Counter:
    def __init__(self, value=0.):
        self.value = value

    def __add__(self, other):
        if isinstance(other, Counter):
            self.value = self.value + other.value
            return self
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            self.value += other
            return self

    def __sub__(self, other):
        if isinstance(other, Counter):
            self.value = self.value - other.value
            return self
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            self.value = self.value - other
            return self

    def __mul__(self, other):
        if isinstance(other, Counter):
            self.value = self.value * other.value
            return self
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            self.value = self.value * other
            return self

    def __truediv__(self, other):
        if isinstance(other, Counter):
            self.value = self.value / other.value
            return self
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            self.value = self.value / other
            return self.value

    def __rmod__(self, other):
        if isinstance(other, Counter):
            return self.value % other.value
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            return self.value % other

    def __lt__(self, other) -> bool:
        if isinstance(other, Counter):
            return self.value < other.value
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            return self.value < other
        else:
            raise TypeError("'<' not supported between instances of '{}' and '{}'".format(str(Counter.__class__),
                                                                                          str(other.__class__)))

    def __le__(self, other) -> bool:
        if isinstance(other, Counter):
            return self.value <= other.value
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            return self.value <= other
        else:
            raise TypeError("'<=' not supported between instances of '{}' and '{}'".format(str(Counter.__class__),
                                                                                           str(other.__class__)))

    def __gt__(self, other) -> bool:
        if isinstance(other, Counter):
            return self.value > other.value
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            return self.value > other
        else:
            raise TypeError("'>' not supported between instances of '{}' and '{}'".format(str(Counter.__class__),
                                                                                          str(other.__class__)))

    def __ge__(self, other) -> bool:
        if isinstance(other, Counter):
            return self.value >= other.value
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
            return self.value >= other
        else:
            raise TypeError("'>=' not supported between instances of '{}' and '{}'".format(str(Counter.__class__),
                                                                                           str(other.__class__)))

    def __str__(self):
        return str(self.value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)


if __name__ == '__main__':
    a = Counter(0)
    c = Counter(0)
    for i in range(10):
        c += i
    c += 5.5
    if c <= 5.5:
        print("---------")
    print(c)
