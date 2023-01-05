import matplotlib.pylab as plt


class InfoPlotter:
    def __init__(self):
        self.info = {}

    def add_info(self, info):
        for key in info:
            if key not in self.info.keys():
                self.info[key] = [info[key]]
            else:
                self.info[key].append(info[key])
        # print("info:", info)

    def plot(self):
        if "step" in self.info.keys():
            x_axis_ = self.info["step"]
            self.info.pop("step")
        else:
            x_axis_ = range(len(self.info[list(self.info.keys())[0]]))

        # plot all the items in self.info
        for key in self.info.keys():
            plt.plot(x_axis_, self.info[key], label=key)

        plt.xlabel('step')
        plt.ylabel('value')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    a = {"real_v": 0, "planned_v": 0, "planned_left_v": 0, "planned_right_v": 0,
         "step": 0}
    for item in a:
        print(item)
