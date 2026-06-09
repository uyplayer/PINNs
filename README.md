# [Physics Informed Neural Networks](https://maziarraissi.github.io/PINNs/)

> **Notice:** This repository is no longer under active maintenance. It is highly recommended to utilize implementations of Physics-Informed Neural Networks (PINNs) available in [PyTorch](https://github.com/rezaakb/pinns-torch), [JAX](https://github.com/rezaakb/pinns-jax), and [TensorFlow v2](https://github.com/rezaakb/pinns-tf2).

We introduce physics informed neural networks – neural networks that are trained to solve supervised learning tasks while respecting any given law of physics described by general nonlinear partial differential equations. We present our developments in the context of solving two main classes of problems: data-driven solution and data-driven discovery of partial differential equations. Depending on the nature and arrangement of the available data, we devise two distinct classes of algorithms, namely continuous time and discrete time models. The resulting neural networks form a new class of data-efficient universal function approximators that naturally encode any underlying physical laws as prior information. In the first part, we demonstrate how these networks can be used to infer solutions to partial differential equations, and obtain physics-informed surrogate models that are fully differentiable with respect to all input coordinates and free parameters. In the second part, we focus on the problem of data-driven discovery of partial differential equations.

For more information, please refer to the following: (<https://maziarraissi.github.io/PINNs/>)

- Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." Journal of Computational Physics 378 (2019): 686-707.

- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)." arXiv preprint arXiv:1711.10561 (2017).

- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10566)." arXiv preprint arXiv:1711.10566 (2017).

## Citation

    @article{raissi2019physics,
      title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
      journal={Journal of Computational Physics},
      volume={378},
      pages={686--707},
      year={2019},
      publisher={Elsevier}
    }

    @article{raissi2017physicsI,
      title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
      journal={arXiv preprint arXiv:1711.10561},
      year={2017}
    }

    @article{raissi2017physicsII,
      title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
      journal={arXiv preprint arXiv:1711.10566},
      year={2017}
    }

# 学习笔记

 

### 核心路径清单

1. **第一步：Schrodinger (练手基础)**
* **目标：** 理解“物理方程怎么写进代码”。
* **TODO：**
* 找到 `net_f` 函数，确认它就是 Schrödinger 方程的程序化体现。
* 运行程序，确保能看到 Loss 下降过程。


* **为什么：** 这是最标准的正向问题，代码最干净。


2. **第二步：Navier-Stokes (工业核心)**
* **目标：** 理解“怎么从数据里挖出物理参数”。
* **TODO：**
* 搜索 `tf.Variable`。找到那些被定义为“可变”的物理系数（如 $\lambda_1, \lambda_2$）。
* 理解它是如何通过拟合观测数据，把这些系数“反向训练”出来的。


* **为什么：** 这是 PINNs 的核心价值所在，也是工业界最关心的能力。


3. **第三步：KdV (进阶模型)**
* **目标：** 理解“如何处理复杂时间演化”。
* **TODO：**
* 找到它使用 Runge-Kutta 步进逻辑的部分。


* **为什么：** 当你面临时间跨度很长的物理仿真时，普通 PINNs 会崩溃，这时候必须用这种分步处理的方法。



---
 

---

 
