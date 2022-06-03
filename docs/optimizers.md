# Optimizers

## Adam
变量规定
- 动量：$v_t$，初始$v_0$全0
- 梯度：$g_t$
- 梯度累加变量$n_t$，初始$n_0$全零
- $\beta_1$超参数建议0.9

动量更新：
$$v_t\leftarrow \beta_1v_{t-1}+(1-\beta_1)g_t$$

对$g_t^2$加权移动平均：
$$n_t\leftarrow \beta_2n_{t-1}+(1-\beta_2)g_t^2$$

在$t$时刻:
$$v_t=(1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}g_i=1-\beta_1^t$$

偏差修正：当$t$较小时，过去各时间步小批量随机梯度权值之和会较小。例如，当$β_1 = 0.9$时， $v_1 = 0.1g_1$ 。为了消除这样的影响，对于任意时间步$t$，我们可以将 $v_t$再除以$1-\beta_1^t$，从而使过去各时间步$g_t$权值和为1.

$$
\left\{
\begin{aligned}
\hat{v}_t & = & \frac{v_t}{1-\beta_1^t} \\
\hat{n}_t & = & \frac{n_t}{1-\beta_2^t} \\
\end{aligned}
\right.
$$

那么($\eta$ 是学习率，$\epsilon$ 防止分母为0)：
$$g'_t\leftarrow\frac{\eta\hat{v}_t}{\sqrt{\hat{n}_t}+\epsilon}$$

最后
$$x_t\leftarrow x_{t-1}-g'_t$$






## Reference
- [1][SGD及其改进优化器](https://zhuanlan.zhihu.com/p/152566066)