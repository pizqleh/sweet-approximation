# sweet-neural-approximation

Spatial audio methods reproduce a desired auditory scene over a region by approximating the sound wave that generated it using an arrangement of a few loudspeakers. Physically-inspired criteria are typically used to find a good approximation. However, an approximation error that may be physically small may still be perceptually significant. The authors recently proposed a method that leverages psycho-acoustic principles to approximate the sound wave by maximizing the area of the sweet spot, i.e., the region where the approximating and target sound waves are perceptually close. The purpose of this repository is to make that method applicable to real world instances by performing an approximation to it leveraging machine learning.

## Model

Consider an array of $n_s$ speakers located at $x_1, \ldots, x_{n_s} \in \mathbb{R}^3$. When the medium is assumed homogeneous and isotropic, and each loudspeaker is modeled as an isotropic point source, the sound wave they generate is

$$
u(t, x) = \sum_{k=1}^{n_s} \frac{\alpha_k(t - c_s^{-1} \|x - x_k\|)}{4 \pi \|x - x_k\|}
$$

where $c_s$ is the speed of sound in the medium, and $\alpha_1, \ldots, \alpha_{n_s}$ are the audio signals of every loudspeaker. In the frequency domain, this is represented as

$$
\widehat{u}(f, x) = \sum_{k=1}^{n_s} \widehat{\alpha}_k(f) \frac{e^{-2\pi i c_s^{-1} f \|x - x_k\|}}{4\pi \|x - x_k\|}
$$

where $\widehat{\alpha}_k$ is the Fourier transform of $\alpha_k$ in time:

$$
\widehat{\alpha}_k(f) := \int_{R} \alpha_k(t) e^{-2\pi i f t}\, dt.
$$

Also, consider a target sound wave produced by a point source, which can be modeled in the frequency domain as:

$$
\widehat{u}_0(f,x) = \widehat{\alpha}_0(f) \frac{e^{-2\pi i c_s^{-1} f \|x - x_0\|}}{4\pi \|x - x_0\|}
$$

Our objective is to approximate as **best as possible** the sound wave $u_0$ with the array of loudspeakers over a bounded domain $\Omega \subset \mathbb{R}^3$. We assume that $\Omega$ contains no sound point sources, i.e., $x_k \notin \overline{\Omega}$, allowing us to avoid the singularities in the equations above.

## Approximation

In Izquierdo Lehmann et al (2022) [https://ieeexplore.ieee.org/document/9829924], the SWEET method was proposed, which aims to find the loudspeakers' signals $\alpha_k$ that maximize the size of the **sweet spot**, i.e., the region where $u$ resembles $u_0$ in a perceptual way. This method, although effective, is computationally expensive as it requires solving a sequence of convex problems. For this reason, we aim to approximate it.

If we fix the loudspeaker positions $x_1, \dots ,x_{n_s}$, from the equation above, the synthesized sound wave $u$ can be encoded by means of the loudspeaker signals $\alpha_k$, $k=1, \dots , n_s$. For the target sound wave, if we do not fix its position, the sound wave $u_0$ can be encoded by means of $x_0$ and its source signal $\alpha_0$. Assume that $\alpha_k \in L^2(\mathbb{R})$, $k=0, \dots , n_s$. Then the SWEET method can be seen as an operator $M: (\mathbb{R}^3, L^2(\mathbb{R})) \rightarrow (L^2(\mathbb{R}))^{n_s}$ whose input is the codification of the desired sound wave $u_0$ and whose output is the codification of the loudspeaker signals $\alpha_k$.

Our objective is to approximate the $M$ operator. To do this, we construct a sample of $n_r$ input-output relations $S = ((x_0,\alpha_0), (\alpha_k)_{k=1}^{n_s})_{\ell=1}^{n_r}$ in order to learn $M$ utilizing neural networks. Here we observe two difficulties: 

1. The input and output elements $\alpha_k$ are infinite dimensional; even after discretization, the dimension of the signals $\alpha_k$ is too high to be processed by the SWEET method in a reasonable amount of time. The sampling process becomes practically unfeasible.
2. By construction, $M$ is a non-linear operator; neither on the spatial component $x_0 \in \mathbb{R}^3$ nor in the source signal $\alpha_0 \in L^2(\mathbb{R})$. In particular, this implies that for a fixed $x_0$, the relation between the source signal and the $k$-th loudspeaker signal $ \alpha_0 \rightarrow M(x_0,\alpha_0)_k $ cannot be expressed as a convolution.

To avoid these limitations, we seek an approximation of $M$ that is linear in the second argument $\alpha_0$. This **approach** trades accuracy for efficiency in the training stage and in the **a posteriori** usage. Instead of enforcing the linearity of the approximation in the neural network architecture, we do it in three steps:

1. Implicitly reducing the complexity of the elements of the sample $S$, considering **pseudo-sinusoidal** elements.
2. Approximating the SWEET method restricted to the **pseudo-sinusoidal** domain.
3. Constructing linear filters from multiple single frequency approximation outcomes.

For the first step, we choose **pseudo-sinusoidal** signals of the form

$$
\widehat{\alpha}_k(f) = a_k\, e^{-(f - f_0)^2 / 2\sigma^2}
$$

for central frequencies in a frequency band $f_0 \in [f_{\text{min}}, f_{\text{max}}] = F$, coefficients $a_k \in \mathbb{C}$, and a fixed spectral localization parameter $\sigma \ll 1$. Note that, as exposed in Izquierdo Lehmann et al (2022), for all practical purposes, $\widehat{\alpha}_k(f) \approx a_k \delta_{f_0}$.

Under this restricted domain, using spatially-dependent amplitudes $a_0 = a_0(x_0)$ and limiting the positions $x_0$ to a bounded region $\Lambda \subseteq \mathbb{R}^3$, the SWEET method can be seen as a non-linear operator $M_s : (\Lambda,F) \rightarrow \mathbb{C}^{n_s}$ whose input is the codification of the desired sound wave $u_0$ and its output the codification of the loudspeaker signals $\alpha_k$ under the above simplifications. Note that now the input is of dimension at most 4 and the output of complex dimension $n_s$. 

For the second step, we use a fully connected network $N : (\Lambda,F) \rightarrow \mathbb{R}^{2n_s}$ to approximate $M_s$. Here we have split the real part and imaginary parts of the output into two real dimensions. We call this network SWEET-Net. 

For the third step, we consider $\alpha_{d,k} \in \mathbb{R}^{n_d}$ as a discretization of the signals $\alpha_k$ for a sampling frequency $f_s$. Then, taking DFT on them, we obtain $\widehat{\alpha}_{d,k} \in \mathbb{C}^{n_d}$, which represents a signal

$$
\sum_{\ell=0}^{n_d-1}\widehat{\alpha}_{d,k}[\ell]\delta_{f_\ell}
$$

with $f_\ell = \ell f_s / n_d$. Then, for each $x_0 \in \Lambda$, we construct the linear filter $H(x_0) \in \mathbb{C}^{n_d}$ given by

$$
H(x_0)_\ell = N(x_0, \ell f_s / n_d) / a_0(x_0).
$$

Therefore, our approximation of $M$ is defined as $M_a : (\mathbb{R}^3, \mathbb{C}^{n_d}) \rightarrow \mathbb{C}^{n_s \times n_d}$:

$$
[M_a(x_0, \widehat{\alpha}_{d,0})]_k = H(x_0) \widehat{\alpha}_{d,0}.
$$
