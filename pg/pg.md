##  PG Algorithm
### 策略梯度推导
**最大回报函数**：用参数化的神经网络表示我们的策略$\pi_{\theta}$，那我们的目标，就可以表示为调整$\theta$，使得期望回报$\mathcal{R}$最大，用公式表示：
$$
    \mathcal{J}(\pi_{\theta}) = \mathbb{E}[\mathcal{R}(\tau)]
$$ 
在上述式子中，$\tau$通常表示一条完整的路径，而最大化问题我们通常可以使用梯度上升法来做（也可以取反使用梯度下降法来实现），在使用梯度上升（下降）法中，参数$\theta$的更新策略为：
$$
    \theta^* = \theta + \alpha \nabla_{\theta} \mathcal{J}(\pi_{\theta})
$$

**策略梯度**：所谓策略梯度，最关键的是求取最大回报函数中的$\mathcal{J}(\pi_{\theta})$的梯度，即$\nabla_{\theta} \mathcal{J}(\pi_{\theta})$，下面是对梯度的表达式的推导：
$$
\begin{equation}
\begin{aligned}
    \nabla_{\theta} \mathcal{J}(\pi_{\theta}) & = \nabla_{\theta} \mathbb{E}_{\tau ~ \pi_{\theta}}[\mathcal{R}(\tau)] \\
    & = \nabla_{\theta} \int_{\tau} P(\tau | \theta) \mathcal{R}(\tau) \\
    & = \int_\tau \nabla_{\theta} P(\tau | \theta) \mathcal{R}(\tau) \\
    & = \int_\tau P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta) \mathcal{R}(\tau) \\
    & = \mathbb{E}[\nabla_{\theta} \log P(\tau | \theta) \mathcal{R}(\tau)]
\end{aligned}
\end{equation}
$$
因此，对$\mathcal{J}(\pi_{\theta})$的梯度其实就是想当于求取$\nabla_{\theta} \log P(\tau | \theta) \mathcal{R}(\tau)$的期望，而关于$P(\tau | \theta)$，又有如下的等式：
$$
\begin{equation}
\begin{aligned}
P(\tau | \theta) & = \rho_0 (s_0) \prod_{t=0}^T P(s_{t+1} | s_t, a_t) \pi_{\theta}(a_t | s_t) \\
\end{aligned}
\end{equation}
$$

等式两边同时取$\log$，得到如下的等式：
$$
\begin{equation}
\begin{aligned}
\log P(\tau | \theta) & = \log \rho_0 (s_0) \prod_{t=0}^T P(s_{t+1} | s_t, a_t) \pi_{\theta}(a_t | s_t) \\

& = \log \rho_0 (s_0)  + \sum_{t=0} ^ T [\log P(s_{t+1} | s_t, a_t) + \log \pi_{\theta}(a_t | s_t)]
\end{aligned}
\end{equation}
$$
因此将(3)式中的结果代入(1)式中的结果，得到
$$
\begin{equation}
\begin{aligned}
\nabla_{\theta} \mathcal{J}(\pi_{\theta}) & =  \mathbb{E}[\nabla_{\theta} \log P(\tau | \theta) \mathcal{R}(\tau)] \\
& =  \mathbb{E}[\nabla_{\theta}  (\log \rho_0 (s_0)  + \sum_{t=0} ^ T [\log P(s_{t+1} | s_t, a_t) + \log \pi_{\theta}(a_t | s_t)] ) \mathcal{R}(\tau)] \\
& = \mathbb{E}_{\tau ~ \pi_{\theta}} [(\sum_{t=0}^T \nabla_{\theta} \log \pi_{\theta}(a_t | s_t)) \mathcal{R}(\tau)] \\
\end{aligned}
\end{equation}
$$
(4)式子的工程实现可以采用蒙特卡罗的思想来求取期望，也就是采样求均值来近似表示期望。假设我们收集到了一系列的$\mathcal{D} = \{\tau_{i}\}_{i=1,2,...,N}$，其中每条轨迹都是agent使用策略$\pi_\theta$与环境交互采样得到的，因此(4)式也可以近似的为：
$$
\nabla_{\theta} \mathcal{J}(\pi_{\theta}) =\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^T \nabla_{\theta} \log \pi_{\theta}(a_t | s_t)) \mathcal{R}(\tau)
$$
