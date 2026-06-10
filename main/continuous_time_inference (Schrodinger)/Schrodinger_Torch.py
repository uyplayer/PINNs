import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pydoe import lhs


class PhysicsInformedNN(nn.Module):
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub, device):
        """
        初始化PhysicsInformedNN
        :param x0: 初始条件 x 坐标
        :param u0: 初始条件 u 值
        :param v0: 初始条件 v 值
        :param tb: 时间边界
        :param X_f: 训练数据
        :param layers: 网络层数
        :param lb: 边界条件 lb
        :param ub: 边界条件 ub
        :param device: 设备
        """

        super(PhysicsInformedNN, self).__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)

        self.x0 = torch.tensor(x0, dtype=torch.float32).to(device)
        self.t0 = torch.zeros_like(self.x0).to(device)
        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
        self.v0 = torch.tensor(v0, dtype=torch.float32).to(device)

        self.t_lb = torch.tensor(tb, dtype=torch.float32).to(device)
        self.t_ub = torch.tensor(tb, dtype=torch.float32).to(device)
        self.x_lb = (torch.zeros_like(self.t_lb) + lb[0]).requires_grad_(True)
        self.x_ub = (torch.zeros_like(self.t_ub) + ub[0]).requires_grad_(True)

        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32).to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32).to(device)

        self.device = device
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        self.optimizer = optim.Adam(list(self.weights) + list(self.biases), lr=1e-3)
        self.iter_count = 0

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init([layers[l], layers[l + 1]], self.device)
            b = nn.Parameter(
                torch.zeros([1, layers[l + 1]], dtype=torch.float32, device=self.device)
            )
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size, device=None):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        if device is None:
            device = self.device
        return nn.Parameter(
            torch.randn([in_dim, out_dim], dtype=torch.float32, device=device)
            * xavier_stddev
        )

    def neural_net(self, X):
        num_layers = len(self.weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = torch.tanh(torch.add(torch.matmul(H, W), b))

        W = self.weights[-1]
        b = self.biases[-1]
        Y = torch.add(torch.matmul(H, W), b)
        return Y

    def net_uv(self, x, t):
        X = torch.cat([x, t], dim=1)
        uv = self.neural_net(X)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    def net_f_uv(self, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)

        u, v = self.net_uv(x, t)

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]

        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

        return f_u, f_v

    def train_step(self):
        self.optimizer.zero_grad()

        # 初始条件损失
        u0_pred, v0_pred = self.net_uv(self.x0, self.t0)
        loss_u0 = torch.mean((self.u0 - u0_pred) ** 2)
        loss_v0 = torch.mean((self.v0 - v0_pred) ** 2)

        # 边界条件损失
        u_lb, v_lb = self.net_uv(self.x_lb, self.t_lb)
        u_ub, v_ub = self.net_uv(self.x_ub, self.t_ub)

        u_lb_x = torch.autograd.grad(
            u_lb, self.x_lb, torch.ones_like(u_lb), create_graph=True
        )[0]
        v_lb_x = torch.autograd.grad(
            v_lb, self.x_lb, torch.ones_like(v_lb), create_graph=True
        )[0]
        u_ub_x = torch.autograd.grad(
            u_ub, self.x_ub, torch.ones_like(u_ub), create_graph=True
        )[0]
        v_ub_x = torch.autograd.grad(
            v_ub, self.x_ub, torch.ones_like(v_ub), create_graph=True
        )[0]

        loss_bc_u = torch.mean((u_lb - u_ub) ** 2)
        loss_bc_v = torch.mean((v_lb - v_ub) ** 2)
        loss_bc_u_x = torch.mean((u_lb_x - u_ub_x) ** 2)
        loss_bc_v_x = torch.mean((v_lb_x - v_ub_x) ** 2)

        # 物理方程损失
        f_u, f_v = self.net_f_uv(self.x_f, self.t_f)
        loss_f_u = torch.mean(f_u**2)
        loss_f_v = torch.mean(f_v**2)

        # 总损失
        loss = (
            loss_u0
            + loss_v0
            + loss_bc_u
            + loss_bc_v
            + loss_bc_u_x
            + loss_bc_v_x
            + loss_f_u
            + loss_f_v
        )

        loss.backward()
        self.optimizer.step()

        self.iter_count += 1
        if self.iter_count % 10 == 0:
            print(
                f"It: {self.iter_count}, Loss: {loss.item():.3e}, "
                f"Loss_u0: {loss_u0.item():.3e}, Loss_bc: {(loss_bc_u + loss_bc_v).item():.3e}, "
                f"Loss_f: {(loss_f_u + loss_f_v).item():.3e}"
            )

        return loss.item()

    def predict(self, X_star):
        x_star = torch.tensor(
            X_star[:, 0:1], dtype=torch.float32, requires_grad=True
        ).to(self.device)
        t_star = torch.tensor(
            X_star[:, 1:2], dtype=torch.float32, requires_grad=True
        ).to(self.device)

        u_pred, v_pred = self.net_uv(x_star, t_star)
        f_u_pred, f_v_pred = self.net_f_uv(x_star, t_star)

        return (
            u_pred.detach().cpu().numpy(),
            v_pred.detach().cpu().numpy(),
            f_u_pred.detach().cpu().numpy(),
            f_v_pred.detach().cpu().numpy(),
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    data = scipy.io.loadmat("../Data/NLS.mat")
    print("文件里实际的变量名有：", data.keys())

    t = data["tt"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = data["uu"]
    # 数据拆分为实部和虚部
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    # 计算强调
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    # 边界条件
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    # 训练样本数
    N0 = 50  # 初始条件样本数
    N_b = 50  # 边界条件样本数
    N_f = 20000  # 方程样本数
    layers = [2, 100, 100, 100, 100, 2]

    #  随机选取 50 个空间点，用于构建初始时刻的约束
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]  # 初始条件空间点
    u0 = Exact_u[idx_x, 0:1]  # 边界条件初始时刻的实部
    v0 = Exact_v[idx_x, 0:1]  # 边界条件初始时刻的虚部

    # 边界条件采样
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]  # 边界条件时间点

    np.random.seed(1234)
    torch.manual_seed(1234)
    #  生成 20,000 个在时空范围内均匀分布的“采样点”，用于监督物理方程
    X_f = lb + (ub - lb) * lhs(2, N_f)

    # 创建模型
    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub, device)

    # 训练
    nIter = 5000
    print(f"Training for {nIter} iterations...")
    for _ in range(nIter):
        model.train_step()

    # 预测
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    # 计算误差
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)

    print(f"\nError u: {error_u:.3e}")
    print(f"Error v: {error_v:.3e}")
    print(f"Error h: {error_h:.3e}")

    # 绘图
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method="cubic")
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method="cubic")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(
        U_pred.T,
        interpolation="nearest",
        cmap="YlGnBu",
        extent=[lb[1], ub[1], lb[0], ub[0]],
        origin="lower",
        aspect="auto",
    )
    axes[0].set_title("$u(t,x)$")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$x$")

    axes[1].imshow(
        V_pred.T,
        interpolation="nearest",
        cmap="YlGnBu",
        extent=[lb[1], ub[1], lb[0], ub[0]],
        origin="lower",
        aspect="auto",
    )
    axes[1].set_title("$v(t,x)$")
    axes[1].set_xlabel("$t$")

    axes[2].imshow(
        H_pred.T,
        interpolation="nearest",
        cmap="YlGnBu",
        extent=[lb[1], ub[1], lb[0], ub[0]],
        origin="lower",
        aspect="auto",
    )
    axes[2].set_title("$|h(t,x)|$")
    axes[2].set_xlabel("$t$")

    plt.tight_layout()
    plt.savefig("./figures/NLS_pytorch.png", dpi=150)
    plt.show()
