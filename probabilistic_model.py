import torch
import numpy as np
import dlib
import time
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        prediction = self.out(h)
        return prediction, h


class rnn_method():
    def __init__(self, buffer_size=5):
        self.position_buffer = np.zeros((buffer_size, 2))
        self.velocity_buffer = np.zeros((buffer_size, 2))
        self.accelerate_buffer = np.zeros((buffer_size, 2))
        self.buffer_count = 0
        self.buffer_size = buffer_size

        self.state_buffer = np.zeros((buffer_size, 2, 2))
        self.state = -1
        self.duration_time = 0
        self.state_count = 0
        self.distance = 700

        self.face = [0, 0, 0]
        self.wait = [0, 0, 0]
        self.time = time.time()
        self.thres = 70

        self.rnn = RNN(4, 10).cuda()
        self.load('models/rnn1.pt')

    def store(self, gaze, face):

        if np.linalg.norm(np.array(face) - np.array(self.face)) > 200:
            if np.linalg.norm(np.array(face) - np.array(self.wait)) < 100:
                self.face = face
                self.wait = [0, 0, 0]
                self.position_buffer = np.zeros((self.buffer_size, 2))
                self.velocity_buffer = np.zeros((self.buffer_size, 2))
                self.accelerate_buffer = np.zeros((self.buffer_size, 2))
                self.buffer_count = 0

            else:
                self.wait = face
                return
        time_tmp = time.time()
        interval = (time_tmp - self.time) / 2
        self.time = time_tmp
        self.wait = [0, 0, 0]

        self.position_buffer[self.buffer_count % self.buffer_size, :] = gaze
        if self.buffer_count > 0:

            velocity = [(gaze[0] - self.position_buffer[(self.buffer_count - 1) % self.buffer_size][0]) / interval,
                        (gaze[1] - self.position_buffer[(self.buffer_count - 1) % self.buffer_size][1]) / interval]
            # print(velocity)
            self.velocity_buffer[self.buffer_count % self.buffer_size, :] = velocity

            if self.buffer_count > 1:
                accelerate = [
                    (velocity[0] - self.velocity_buffer[(self.buffer_count - 1) % self.buffer_size][0]) / interval,
                    (velocity[1] - self.velocity_buffer[(self.buffer_count - 1) % self.buffer_size][1]) / interval]
                self.accelerate_buffer[self.buffer_count % self.buffer_size, :] = accelerate

        self.buffer_count += 1

    def analysis(self):
        if self.buffer_count < 6 or self.wait != [0, 0, 0]:
            return -1, 0, 0
        seq = np.zeros((1, 5, 4))
        index = self.buffer_count % self.buffer_size
        seq[0, 0:self.buffer_size - index, 0:2] = self.position_buffer[index:self.buffer_size, :]
        seq[0, self.buffer_size - index:self.buffer_size, 0:2] = self.position_buffer[0:index, :]
        seq[0, 0:self.buffer_size - index, 2:4] = self.velocity_buffer[index:self.buffer_size, :]
        seq[0, self.buffer_size - index:self.buffer_size, 2:4] = self.velocity_buffer[0:index, :]
        with torch.no_grad():
            input = torch.from_numpy(seq).float().cuda()
            h = torch.zeros([1, 1, 10]).float().cuda()
            pred, h = self.rnn(input, h)
            pred = F.sigmoid(pred)
            prediction = pred.cpu().detach().numpy()

        return prediction[0][0][0] > 0.5, np.mean(self.position_buffer, axis=0), np.mean(self.velocity_buffer, axis=0)

    def load(self, file):
        self.rnn.load_state_dict(torch.load(file))
        self.rnn.eval()


class thres_method():
    def __init__(self, buffer_size=5):
        self.position_buffer = np.zeros((buffer_size, 3))
        self.velocity_buffer = np.zeros((buffer_size, 2))
        self.accelerate_buffer = np.zeros((buffer_size, 2))
        self.buffer_count = 0
        self.buffer_size = buffer_size

        self.state_buffer = np.zeros((buffer_size, 2, 2))
        self.state = -1
        self.duration_time = 0
        self.state_count = 0

        self.face = [0, 0, 0]
        self.wait = [0, 0, 0]
        self.time = time.time()
        self.thres = 70

    def store(self, gaze, face):

        if np.linalg.norm(np.array(face) - np.array(self.face)) > 200:
            if np.linalg.norm(np.array(face) - np.array(self.wait)) < 100:
                self.face = face
                self.wait = [0, 0, 0]
                self.position_buffer = np.zeros((buffer_size, 2))
                self.velocity_buffer = np.zeros((buffer_size, 2))
                self.accelerate_buffer = np.zeros((buffer_size, 2))
                self.buffer_count = 0

            else:
                self.wait = face
                return
        time_tmp = time.time()
        interval = time_tmp - self.time
        self.time = time_tmp
        self.wait = [0, 0, 0]
        self.position_buffer[self.buffer_count % self.buffer_size, :] = gaze
        if self.buffer_count > 0:

            velocity = [(gaze[0] - self.position_buffer[(self.buffer_count - 1) % self.buffer_size][0]) / interval,
                        (gaze[1] - self.position_buffer[(self.buffer_count - 1) % self.buffer_size][1]) / interval]

            self.velocity_buffer[self.buffer_count % self.buffer_size, :] = velocity

            if self.buffer_count > 1:
                accelerate = [
                    (velocity[0] - self.velocity_buffer[(self.buffer_count - 1) % self.buffer_size][0]) / interval,
                    (velocity[1] - self.velocity_buffer[(self.buffer_count - 1) % self.buffer_size][1]) / interval]
                self.accelerate_buffer[self.buffer_count % self.buffer_size, :] = accelerate

        self.buffer_count += 1

    def analysis(self):
        if self.buffer_count < 6 or self.wait != [0, 0, 0]:
            return -1, 0, 0

        total_veloticy = np.sqrt(
            self.velocity_buffer[self.buffer_count % self.buffer_size, 0] ** 2 + self.velocity_buffer[
                self.buffer_count % self.buffer_size, 1] ** 2)
        if total_velocity < self.thres:
            return 0, np.mean(self.position_buffer, axis=0), np.mean(self.velocity_buffer, axis=0)
        else:
            return 1, np.mean(self.position_buffer, axis=0), np.mean(self.velocity_buffer, axis=0)
