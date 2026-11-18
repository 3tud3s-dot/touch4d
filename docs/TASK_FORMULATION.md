## v0 — Gaussian Dynamics Prediction

$$
(GS_t, A_{t:t+4}, T_{t:t+4}) \xrightarrow{\text{Encoder}} Z \xrightarrow{\text{Attention Dynamics}} \hat{Z}_{t+1:t+5} \xrightarrow{\text{Decoder}} \widehat{GS}_{t+1:t+5}
$$

一次性预测未来 5 帧。

## v1 — Fast–Slow Manipulation System

### Slow — Perception @1Hz

$$
H_{t-49:t} \xrightarrow{\text{GS Reconstruction}} GS_t \xrightarrow{\text{Encoder}} Z_t
$$

其中 $H$ 为历史 1 秒的 50 帧数据。

### Latent Dynamics — 补全 49 帧

$$
(Z_t, A, T) \xrightarrow{\text{v0 Dynamics}} \hat{Z}_{t+1:t+49}
$$

在 latent space 补全，不必解码为 GS。具体预测方式待定。

### Fast — Control @50Hz

$$
(\hat{Z}_\tau, C_\tau) \xrightarrow{\text{Fast Controller}} A_\tau^{EE}
$$

其中 $C_\tau$ 为历史 Context，$A_\tau^{EE}$ 为末端执行器动作。

## v2 — Tac 闭环
