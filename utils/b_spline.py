import numpy as np
import matplotlib.pyplot as plt


# Code implemented from: https://opensourc.es/blog/b-spline/


class BSpline:
    def __init__(self):
        # degree
        self.curve = []
        self.ctr_pts = []

    def _generate_knot_vector(self, ctr_num, k):
        # knot vector length = ctr_num + k
        left = [0] * (k + 1)
        inner = [i for i in range(1, ctr_num - k)]
        right = [ctr_num - k] * (k + 1)
        knot_vector = np.array(left + inner + right)
        return knot_vector

    def _b_spline_basis(self, i, k, u, U):
        if k == 0:
            if U[i] <= u < U[i + 1]:
                return 1
            else:
                return 0
        else:
            left = 0
            right = 0
            if U[i + k] - U[i] != 0:
                left = (u - U[i]) / (U[i + k] - U[i]) * self._b_spline_basis(i, k - 1, u, U)
            if U[i + k + 1] - U[i + 1] != 0:
                right = (U[i + k + 1] - u) / (U[i + k + 1] - U[i + 1]) * self._b_spline_basis(i + 1, k - 1, u, U)
            return left + right

    def b_spline(self, ctr_pts, k=2, step_size=100):
        assert len(ctr_pts) > k
        self.ctr_pts = ctr_pts
        self.curve = []

        xx = ctr_pts[:, 0]
        yy = ctr_pts[:, 1]
        ctr_num = len(ctr_pts)
        knot_vector = self._generate_knot_vector(ctr_num, k)
        tt = np.linspace(min(knot_vector), max(knot_vector), step_size)
        tt = tt[0:-1]

        for ind, t in enumerate(tt):
            # determine the interval for this t
            start = 0
            pos_x = 0
            pos_y = 0
            for i in reversed(range(len(knot_vector))):
                if t >= knot_vector[i]:
                    start = i + 1
                    break
            for i in range(start - k - 1, start):
                pos_x += xx[i] * self._b_spline_basis(i, k, t, knot_vector)
                pos_y += yy[i] * self._b_spline_basis(i, k, t, knot_vector)

            self.curve.append([pos_x, pos_y])

        self.curve.append([*ctr_pts[-1]])
        self.curve = np.array(self.curve)
        return self

    def get_deformed_points(self):
        index = np.linspace(0, len(self.curve) - 1, len(self.ctr_pts))
        index = index.astype(np.int)
        return self.curve[index]

    def plot(self):
        dBx = self.ctr_pts[:, 0]
        dBy = self.ctr_pts[:, 1]
        curve_x = self.curve[:, 0]
        curve_y = self.curve[:, 1]
        deformed_points = self.get_deformed_points()

        plt.scatter(dBx, dBy, c='r')
        plt.plot(dBx, dBy, 'r-')

        plt.scatter(curve_x, curve_y, c='b')
        plt.plot(curve_x, curve_y, c='b')

        plt.scatter(deformed_points[:, 0], deformed_points[:, 1], c='g')
        plt.show()


if __name__ == '__main__':
    # dBx = [0, 0, 1, 3, 5, 6, 6, 5, 3, 1, 0, 0]
    # dBy = [3, 1, 3, 0, 1, 1, 5, 6, 6, 6, 5, 3]
    dBx = [0, 0, 1, 3]
    dBy = [3, 1, 3, 0]
    ctr_pts = np.array([dBx, dBy]).transpose([1, 0])
    degree = 2
    step_size = 200
    spline = BSpline()
    spline.b_spline(ctr_pts, degree, step_size).plot()

    print()
