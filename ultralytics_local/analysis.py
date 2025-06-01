import io
import matplotlib.pyplot as plot
import numpy as np
import csv


def read(path):
    rf0 = io.open(path, "r")
    r0 = csv.reader(rf0)
    rd0 = np.zeros((0, 2))
    for row in r0:
        if row[0] != 'epoch':
            arr = [float(row[0]), (float(row[2]) * 7.5 + float(row[3]) * 0.5 + float(row[4]) * 1.5) / 9.5]
            arr = np.array(arr)
            arr.resize((1, 2))
            rd0 = np.append(rd0, arr, axis=0)
    rf0.close()
    return rd0


rd0 = read("results.csv")
rd1 = read("results (1).csv")
rd2 = read("results (2).csv")

plot.figure(1)
plot.plot(rd0[:, 1], label="yoloV11Vit")
plot.plot(rd2[:, 1], label="yoloV11VitFast")
plot.plot(rd1[:, 1], label="yoloV11(Base)")
plot.xlim(10, 50)
plot.ylim(1.15, 1.7)
plot.title("Weighted Loss")
plot.legend()
plot.show()
