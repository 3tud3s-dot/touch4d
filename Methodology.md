# Methodology — G5-Direct

Touch4D predicts cloth motion from an initial 3D Gaussian reconstruction, gripper motion and tactile observations. The demo uses **G5-Direct E8** (620,511 parameters), implemented as [ClothDynamicsPredictor](scripts/training/cloth_model.py). E8 was trained on Episode 0 of Deform360's 008-pink-cloth scene for eight epochs of 140 windows each, totaling 1,120 optimizer updates. The rollout horizon is 15 consecutive frames with a stride of one frame. See the [training and evaluation protocol](configs/workflow.json).

## 1. Input and rollout state

Index time relative to the start of a window. Let $X_0 \in \mathbb R^{N\times3}$ be the observed Gaussian centers and $A_0\in\mathbb R^{N\times11}$ the associated attributes. Each Gaussian has 14 features.

| Attribute | Dimensions | Rollout behavior |
|---|---:|---|
| Center (XYZ) | 3 | Predicted at every step |
| Log scale | 3 | Inherited from the first frame |
| Rotation quaternion (WXYZ) | 4 | Inherited from the first frame |
| Opacity logit | 1 | Inherited from the first frame |
| Spherical-harmonic DC color | 3 | Inherited from the first frame |

Step $k$ predicts frame $k$ using the previous predicted geometry and robot conditions through frame $k-1$. Gaussian identities are retained from initialization. The reference Gaussian sets are independently reconstructed per frame and can differ in count and order. The experiment uses fixed training-derived normalization for both the object and robot coordinates. See the [data protocol](docs/DATASET_PROTOCOL.md) and [data preparation code](scripts/training/cloth_data.py).

## 2. Gaussian encoding and anchor queries

At the start of each window, farthest-point sampling selects up to 128 valid Gaussians as anchors. The model fixes the 16-nearest-neighbor graph, 16-point anchor pooling neighborhoods and radial basis function (RBF) interpolation weights from the initial geometry. Anchor indices remain fixed and their positions follow the predicted Gaussian centers.

At each step, a shared MLP encodes the current Gaussian state. A four-head local attention block aggregates features over the fixed graph using relative positions computed from the current predicted geometry. Mean pooling over each anchor's initial neighborhood and a learned prediction-step embedding produce the query tensor.

$$
F_{k-1}=\operatorname{LocalGSBlock}\left(
    \operatorname{MLP}_{14\to128\to128}([\widehat X_{k-1},A_0]),
    \widehat X_{k-1}
\right), \qquad
Q_k^{(0)}=\operatorname{Pool}_0(F_{k-1})+e^{\mathrm{pred}}_k .
$$

The query tensor has shape $B\times128\times128$ with padding masked throughout. See the [model](scripts/training/cloth_model.py) and [anchor field](scripts/training/gs_decoder.py).

## 3. Gripper and tactile conditioning

Each potential hand has one hand token and 30 controller-point tokens. Separate shared MLPs encode the two types of input, each with widths of $13\to128\to128$.

| Token | Input |
|---|---|
| Hand | Root position (3), rotation matrix (9), gripper opening (1) |
| Controller point | Current XYZ (3), signed XYZ displacement from the initial condition (3), a validity flag for both endpoints (1), and six tactile features (6) |

The tactile features come from four $12\times32$ arrays aligned to the observation time. For each controller point, the features include a Gaussian-weighted mean, a local maximum, coverage, normalized sample age, and validity flags for the mean and maximum. The mean uses a 10 mm kernel in sensor-local coordinates. Valid samples must be at most 75 ms old. Masks distinguish absent hands, invalid controller points and missing tactile samples.

The encoders add identity and condition-time embeddings. Each hand token is then projected and fused into its 30 controller tokens.

$$
\widetilde C_{j,k}=\operatorname{LayerNorm}
\left(C_{j,k}+W_H H_{h(j),k}\right).
$$

The resulting 60 controller tokens serve as keys and values. Separate hand tokens also provide context for the Global and Local motion branches. See the [condition encoder](scripts/training/cloth_conditions.py).

## 4. Geometry-aware cross-attention

Two blocks update the anchor queries through cross-attention, each with four heads of width 32. For a current predicted anchor $p_a$ and an observed controller point $c_j$, define

$$
g_{aj}=\left[\frac{p_a-c_j}{\sigma_0},
             \frac{\lVert p_a-c_j\rVert^2}{\sigma_0^2}\right].
$$

The bandwidth $\sigma_0$ is computed from the initial anchors. Each head $\ell$ combines a dot product, a linear geometry score and a learned pairwise correction.

$$
s_{aj}^{(\ell)}
=\frac{(q_a^{(\ell)})^\top k_j^{(\ell)}}{\sqrt{32}}
+(w_g^{(\ell)})^\top g_{aj}
+f_\ell([q_a^{(\ell)},k_j^{(\ell)},g_{aj}]).
$$

The pairwise MLP has widths of $68\to32\to1$. The geometric weights start at $[0,0,0,-0.5]$. Masked softmax covers valid controllers and a null option with zero logit and zero value. When no controller is valid, the attention contribution is zero. Output projection, residual connections and a feed-forward network update the anchor queries. The Base motion head also uses the final block's attention weights. See the [cross-attention block](scripts/training/cloth_attention.py).

## 5. Base, Global and Local motion

The three branches jointly predict motion in the anchor control space. Let $U_{a,k}$ denote the final query for anchor $a$.

### Base motion

The Base MLP has widths of $151\to128\to3$ and produces a cumulative displacement $d^B_{a,k}$ at each anchor.

| Input | Dimensions |
|---|---:|
| Final anchor query $U_{a,k}$ | 128 |
| Adjacent-frame controller displacements pooled independently by each attention head | 12 |
| Adjacent predicted anchor displacement | 3 |
| Previous cumulative displacement after combining all three branches | 3 |
| Attention mass on controller points valid at both adjacent times, one scalar per head | 4 |
| Anchor history validity flag | 1 |

Each head retains its own XYZ displacement. Pooling uses the controller portion of the original attention distribution, with the null mass still accounted for in its normalization. The first step uses zero motion-history features. See the [Base inputs](scripts/training/cloth_model.py#L86).

### Global translation

For each hand $h$, the model constructs a 12-dimensional signed basis from four XYZ vectors: root motion since the initial condition, root motion since the previous condition, mean cumulative controller motion and mean adjacent controller motion. Each controller mean considers points valid at the two relevant times.

$$
b^G_{h,k}=\left[
\Delta_0 r_h,\ \Delta_1 r_h,\
\operatorname{mean}_{j}(\Delta_0 c_{h,j}),\
\operatorname{mean}_{j}(\Delta_1 c_{h,j})
\right].
$$

A shared $268\to128\to36$ MLP takes the mean valid anchor query (128), the hand token (128) and this basis (12). It produces a $3\times12$ matrix for each hand. With hand validity $m_{h,k}$, the sum of mapped contributions is shared by all anchors.

$$
t_{h,k}=M^G_{h,k}b^G_{h,k},\qquad
g_k=\sum_h m_{h,k}t_{h,k}.
$$

### Local correction

For each anchor and hand, the model separately pools cumulative and adjacent controller displacements. The distance kernel uses current predicted anchor positions and observed controller positions. Each softmax is normalized within the hand over controller points valid at both endpoints of the corresponding displacement. An empty set contributes zero.

$$
\beta^{(i)}_{a,h,j,k}
=\operatorname{MaskedSoftmax}_j
\left(-\frac{\lVert p_a-c_{h,j}\rVert^2}{2\sigma_0^2}\right),
\qquad
u^{(i)}_{a,h,k}=\sum_j\beta^{(i)}_{a,h,j,k}\Delta_i c_{h,j},
\quad i\in\{0,1\}.
$$

The two pooled vectors extend the hand's global basis to $b^L_{a,h,k}=[b^G_{h,k},u^{(0)}_{a,h,k},u^{(1)}_{a,h,k}]\in\mathbb R^{18}$. These geometric weights are computed separately from cross-attention and have no hard distance cutoff.

| Local mapping input | Dimensions |
|---|---:|
| Anchor query and hand token | $128+128$ |
| Weighted relative XYZ and mean distance in units of $\sigma_0$ | 4 |
| Signed motion basis | 18 |
| Anchor history, previous cumulative displacement and the hand's Global contribution | $3+3+3$ |

A shared $287\to128\to54$ MLP produces a $3\times18$ matrix for each anchor and hand. Applying it to the basis and summing valid hands yields

$$
d^L_{a,k}=\sum_h m_{h,k}M^L_{a,h,k}b^L_{a,h,k}.
$$

The Local branch can produce a nonzero spatial mean, allowing it to adjust translation as well as relative shape. Each anchor's correction is applied directly. See the [Global and Local mappings](scripts/training/cloth_motion.py).

## 6. Fixed RBF decoding and recurrence

The three motion contributions combine before interpolation. Each row of $W_0$ is a normalized Gaussian weight over the valid initial anchors. The bandwidth is based on the median nearest-other-anchor distance in the initial geometry.

$$
d_{a,k}=d^B_{a,k}+g_k+d^L_{a,k},
\qquad
W_{0,ia}=\operatorname{MaskedSoftmax}_{a}
\left(-\frac{\lVert x_{i,0}-p_{a,0}\rVert^2}{2\sigma_0^2}\right).
$$

$$
\boxed{\widehat X_k=X_0+W_0D_k}
$$

Here $D_k$ stacks the **cumulative** anchor displacements relative to the initial state. The next step re-encodes $[\widehat X_k,A_0]$ and uses $D_k$, the recent predicted anchor positions and the previous observed controller positions to compute motion history. After the initial observation, all object states come from the model's own predictions.

At H1, the Global and Local bases are zero because the available condition is the window's initial observation. Base provides the first prediction. The terminal layers of all three motion MLPs start at zero, so a freshly initialized model reproduces the initial Gaussian state throughout the rollout. See the [recurrence](scripts/training/cloth_model.py#L66).

## 7. Training objective

Training unrolls all 15 steps and backpropagates through the predicted states. The objective combines a geometric loss with multi-view rendering losses.

$$
\mathcal L=\mathcal L_{\mathrm{CD}}+\mathcal L_{\mathrm{RGB}}+0.1\,\mathcal L_{\alpha}.
$$

The geometric loss averages the two directional unsquared nearest-neighbor distances between predicted and reference centers in normalized coordinates. The rendering losses compare RGB and alpha from the same cameras using mean absolute error in the target-depth-valid region. The rendering targets come from the reference Gaussian reconstructions. See the [training implementation](scripts/training/cloth_train.py) and the [original architecture notes](docs/TOUCH4D_V4_STAGE1_ARCHITECTURE_DRAFT.md#touch4d-g5-directh15-核心计算图).
