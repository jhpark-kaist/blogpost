---
layout: distill
title: AI810 Blog Post (20248084)
description: Generative models traditionally operate in Euclidean space. However, many types of data are naturally manifold-valued, meaning they live on curved or constrained spaces rather than a flat plane. In this article, we explore Geometric Flow Matching (GFM) an emerging framework that adapts flow-based generative modeling to work natively on manifolds. We will build intuition for why this is needed (especially for protein structures), review how standard Flow Matching works in Euclidean settings, and then explain the manifold generalization. Along the way, we use analogies, minimal math, and visual aids to keep things intuitive yet rigorous.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# anonymize when submitting 
authors:
  - name: Joonhyeong Park
    affiliations:
      name: KAIST

bibliography: 2025-04-28-geometric-flow-matching.bib

toc:
  - name: Why Generative Models on Manifolds?
  - name: Recap - Flow Matching in Euclidean Space
  - name: Preliminaries - Manifolds
  - name: How Geometric Flow Matching Works
  - name: Case Study - Protein Modeling with GFM
  - name: Conclusion

---

## Why Generative Models on Manifolds?

Real-world data often have inherent geometric constraints that make their natural domain a manifold instead of a full Euclidean space. **Proteins** provide an excellent example: a protein’s 3-D conformation can be described by torsion angles (like phi/psi backbone angles or side-chain rotations), which are periodic variables on a circle ($$S^1$$). Similarly, the overall orientation and position of a protein in space lie in the $$\mathrm{SE}(3)$$ group of 3-D rotations and translations – another manifold. If we naively treat such data as unconstrained vectors in $$\,\mathbb{R}^n\,$$, we risk modeling artifacts:

* **Periodic Angles** – Imagine modeling a dihedral angle that wraps around at $$360^\circ$$. In Euclidean representation, $$0^\circ$$ and $$360^\circ$$ appear far apart, even though they are physically the same conformation. A Euclidean generative model might struggle, placing undue probability mass at the artificial boundaries (e.g. around $$0^\circ$$ vs $$360^\circ$$) because it doesn’t “know” the variable is circular.  
* **Rotations** – A protein domain’s orientation is a point on the sphere of rotations (technically $$\mathrm{SO}(3)$$). If we parameterize orientation by Euler angles in $$\,\mathbb{R}^3\,$$, the model can suffer from discontinuities (gimbal lock, etc.) and non-uniform coverage of orientation space. For instance, sampling each Euler angle from a Gaussian does not yield a uniform random rotation – the distribution gets warped by the coordinate parameterization.  
* **Rigid Motions** – When placing multiple protein subunits relative to each other, their relative pose is in $$\mathrm{SE}(3)$$. Treating this as six independent numbers (three for translation, three for rotation) in $$\,\mathbb{R}^6\,$$ ignores the fact that rotations have a curved geometry. A naive model might generate “average” rotation matrices that are not valid rotations, or concentrate samples in certain regions of orientation space.  

$$
\begin{align*}
S^1 &:\ \text{unit circle (torsion angle domain)} \\
\mathrm{SO}(3) &:\ \text{3D rotations manifold} \\
\mathrm{SE}(3) &:\ \text{3D rotation + translation group} \\
\mathbb{R}^n &:\ \text{Euclidean space (flattened representation)} \\
\mathbb{R}^6 &:\ \text{naive 3D translation + rotation encoding}
\end{align*}
$$

These issues hint that to model such manifold-valued data, we should respect the geometry. Indeed, machine-learning researchers have noted that many observations in vision, graphics, and biology are non-Euclidean, and purely Euclidean generative techniques are inadequate <d-cite key="zhen2021flow"></d-cite>. For proteins, recent successes in generative modeling (like diffusion models for protein structures) had to incorporate geometric awareness. For example, the **Proteus** diffusion model built in triangle geometry to generate protein backbones without a pre-trained guide <d-cite key="wang2024proteus"></d-cite>. In general, a generative model “aware” of the manifold can sample more valid and realistic data by avoiding distortions that arise from flattening the manifold incorrectly.  

### Why not just embed the manifold in $$\mathbb{R}^n$$?  

One might think we can bypass the issue by representing manifold data in some Euclidean coordinates (for example, using 3-D Cartesian coordinates for points on a sphere). However, while you can represent any manifold data in $$\mathbb{R}^n$$, the distribution often becomes complicated in those coordinates. It might concentrate on a lower-dimensional surface in $$\mathbb{R}^n$$ or have discontinuous support. Generating new samples in the ambient space will likely produce points off the manifold (e.g. points not exactly on the sphere or violating constraints). Enforcing the manifold constraints after generation can be non-trivial or can introduce bias. Hence, there is strong motivation to build generative models that operate directly on the manifold, ensuring all samples and intermediate states respect the underlying geometric constraints.

$$
\begin{align*}
0^\circ &\sim 360^\circ\quad \text{(identical torsion angles)} \\
\text{Sampling in } \mathbb{R}^n &\Rightarrow \text{ off-manifold points or invalid constraints}
\end{align*}
$$

Protein modeling epitomizes this need. A protein backbone is essentially a series of bond angles and lengths – many of which (bond lengths, planar angles) are fixed, leaving torsion angles that rotate freely but periodically. Traditional generative models that ignore these periodicities might output physically implausible structures or require ad-hoc fixes (like wrapping angles or rejecting invalid outputs). By developing generative models on manifolds, researchers aim to produce proteins natively in valid conformation spaces (like torsion-angle space or $$\mathrm{SE}(3)$$ space) without falling off the manifold. The same principle applies to other domains (robotics with rotation trajectories, climate data on Earth’s sphere, etc.), but we’ll keep our focus on proteins as a running example.


---

## Recap - Flow Matching in Euclidean Space

Before diving into manifold techniques, let’s briefly review **Flow Matching (FM)** in the standard Euclidean setting. Flow Matching is a relatively new paradigm for training *continuous normalizing flows* (CNFs) – a class of generative models where you transform a simple distribution (like Gaussian noise) into the data distribution by integrating an ODE (ordinary differential equation). Traditionally, CNFs are trained by maximum likelihood, which involves computing Jacobian determinants or simulating differential equations, which can be expensive and unstable. Flow Matching takes a different route: **don’t simulate the entire flow during training – instead, directly match the velocity field**<d-cite key="lipman2022flow"></d-cite>.

In Flow Matching, one first defines a smooth path of probability distributions $$p_t$$ for $$t\in[0,1]$$ that connects the known source distribution $$p_{t=0}$$ (usually something simple like $$\mathcal{N}(0,I)$$ noise) to the target data distribution $$p_{t=1}$$ (the distribution of your dataset). Think of this path as a plan for how a blob of probability mass should morph from noise into data. A common choice is a Gaussian diffusion path (like gradually adding noise to data or vice versa), but other choices are possible (e.g. an optimal-transport interpolation). Flow Matching doesn’t commit to a specific simulation process; it just needs a prescribed $$p_t$$.

The goal is to find a time-dependent vector field $$v(t,x)$$ such that if we move particles according to the ODE:

$$
\begin{align*}
\frac{d\mathbf{x}}{dt} = v(t,\mathbf{x})
\end{align*}
$$

it will carry out the transformation from:

$$
\begin{align*}
p_0 \longrightarrow p_1\quad \text{following the path } p_t
\end{align*}
$$

In other words, $$v(t,x)$$ should be the ideal velocity field that transports the distribution along the chosen path. If we knew this ideal $$v$$, we could generate samples by integrating:

$$
\begin{align*}
\mathbf{x}(0) \sim p_0 \quad \Rightarrow \quad \mathbf{x}(1) \sim p_1
\end{align*}
$$

Flow Matching gives a clever way to train a neural network to approximate $$v(t,x)$$ **without ever simulating the ODE during training**. Instead, one samples pairs $$(t,x_t,x_1)$$ where:

$$
\begin{align*}
x_t \sim p_t,\quad x_1 \sim p_1
\end{align*}
$$

and then the network is asked to predict the “target” velocity $$u(t,x_t)$$ that would point $$x_t$$ toward a sample from the later distribution. In practice, one can derive an analytical form for this target velocity under many paths. For example, with a diffusion (Gaussian) interpolation, the target vector field is proportional to the *score* (gradient of log-density) or the difference between $$x_t$$ and $$x_1$$ in certain cases.

The training objective is a regression loss:

$$
\begin{align*}
v_\theta(t,x_t) \approx u(t,x_t)
\end{align*}
$$

so we minimize:

$$
\begin{align*}
\mathcal{L} = \mathbb{E}\left[ \left\| v_\theta(t, x_t) - u(t, x_t) \right\|^2 \right]
\end{align*}
$$

This way, the model “matches” the flow velocities, hence the name. Crucially, this procedure is simulation-free – we don’t have to integrate the ODE in inner loops; we only need to sample from $$p_t$$ and $$p_1$$, which is often easy by construction.

Flow Matching has notable advantages. By avoiding back-propagation through ODE solvers or computing log-determinants, it **scales to large problems and tends to be more stable**. In fact, using diffusion paths, Flow Matching was shown to train diffusion models more robustly than the usual score-matching approach. It also allows exploring non-diffusive paths: Lipman *et al.* demonstrated that using an optimal-transport (OT) path (which linearly interpolates distributions in a mass-preserving way) yields faster convergence and better samples than standard diffusion. In summary, in Euclidean space FM provides a flexible framework to train generative flows by “teaching” a vector field how to morph one distribution into another, rather than simulating it blindly.

**Limitations:** Flow Matching as described assumes we can freely sample and compute differences in the ambient space $$\mathbb{R}^n$$. If our data lies on a manifold, applying Euclidean FM directly can be problematic. For instance, if $$x_t$$ and $$x_1$$ are points on a sphere or in $$\mathrm{SO}(3)$$, subtracting them or mixing them linearly is not geometrically meaningful (you can’t just average two rotations by component-wise linear interpolation without leaving $$\mathrm{SO}(3)$$). We need to reformulate the idea of a “conditional path” and “velocity field” in a way that lives on the manifold. **This is where Geometric Flow Matching enters the scene.**

---

## Preliminaries - Manifolds

Before tackling flow matching on manifolds, let’s ensure we understand some basics of manifold geometry.

A **manifold** is a space that may be curved or constrained, but locally it looks like a flat Euclidean space. Classic examples include the surface of a sphere, a torus (doughnut shape), or the rotation group $$\mathrm{SO}(3)$$ (which topologically is like a 3-D sphere with opposite points identified). Formally, a $$d$$-dimensional manifold $$M$$ is something that can be covered by coordinate patches (called *charts*), each of which is like a mapping from an open set of $$\,\mathbb{R}^d$$ to $$M$$. However, no single flat chart can cover the entire manifold without distortion or singularities, much like no flat map can perfectly represent the entire Earth.

**Analogy (Earth Maps):** Think of the Earth (a sphere) as our manifold. We can define latitude and longitude as coordinates – that’s a chart covering most of the globe. But it breaks down at the poles (longitude is undefined at exactly the North Pole). We might use another chart for polar regions. Each chart is like a different “map projection” – Mercator, polar projection, etc. They all overlap partially and relate via coordinate transformations. So, working on a manifold often means you either restrict yourself to a chart (and remember to switch charts when needed), or you do calculations in a coordinate-free way using geometric constructs.

One key construct is the **tangent space**. At any point $$p$$ on the manifold, the tangent space $$T_pM$$ is a flat $$d$$-dimensional space consisting of the possible “directions” one can move infinitesimally from $$p$$ on the manifold. If you’ve seen a tangent line to a curve or a tangent plane to a surface, that’s the idea – it “just touches” the manifold at that point and is the best linear approximation of the manifold near $$p$$. For a sphere, the tangent space at a point is the plane tangent to the sphere at that point.

> A sphere with a tangent plane at point P. The plane represents the tangent space $$T_PM$$, a local linear approximation of the manifold. You can move in any direction along the tangent plane, which corresponds to moving along the sphere’s surface (locally) in some direction.

Why do we care about tangent spaces? Because calculus and differential equations on manifolds are formulated in terms of tangent vectors. A velocity or direction of motion at a point on the manifold lives in the tangent space. If we want to define a vector field $$v(t,x)$$ on a manifold, we must ensure:

$$
\begin{align*}
v(t,x) \in T_xM
\end{align*}
$$

(it points along the manifold, not poking out of it).

Another essential pair of tools are the **exponential map** and **logarithmic map** (exp and log for short). These maps help us go back-and-forth between the manifold and tangent spaces in a well-defined way:

* **Log map $$\log_p(q)$$** – Given two points $$p,q\in M$$, the log map at $$p$$ yields a tangent vector $$v\in T_pM$$ which points from $$p$$ in the direction of $$q$$, with a length equal to the geodesic distance from $$p$$ to $$q$$. In intuitive terms, $$\log_p(q)$$ answers: “if I start at $$p$$ and walk toward $$q$$ along the shortest path on the manifold, what direction and distance should I go initially?”  
* **Exp map $$\exp_p(v)$$** – This is (locally) the inverse of log. Given a tangent vector $$v\in T_pM$$, $$\exp_p(v)$$ gives the point $$q$$ on the manifold you reach by starting at $$p$$ and moving in the direction of $$v$$ for a unit time (or until you cover the distance $$|v|$$).  

$$
\begin{align*}
\log_p(q) &\in T_pM \\
\exp_p(v) &\in M \quad \text{(for } v \in T_pM \text{)}
\end{align*}
$$

On a Euclidean plane:

$$
\begin{align*}
\exp_p(v) &= p + v \\
\log_p(q) &= q - p
\end{align*}
$$

But on a sphere, these operations involve trigonometry (great-circle arcs). On $$\mathrm{SO}(3)$$, $$\exp$$ and $$\log$$ can be related to matrix exponentials and rotation vectors.

The exp/log maps give a way to translate problems on a manifold into problems in a tangent (Euclidean) space, do some computation, and perhaps map back. For instance, to interpolate between two manifold points $$p$$ and $$q$$, a natural path is the **geodesic**:

$$
\begin{align*}
\gamma(t) = \exp_p\left(t \cdot \log_p(q)\right), \quad 0 \le t \le 1
\end{align*}
$$

This yields a curve that starts at $$p$$ ($$t=0$$) and ends at $$q$$ ($$t=1$$).

Finally, if we do embed the manifold in a higher-dimensional Euclidean space (like a sphere in $$\mathbb{R}^3$$), we can talk about **projecting** vectors onto the tangent space. For example, if you have some ambient vector in $$\mathbb{R}^3$$ at point $$p$$ on the sphere, the tangent projection would subtract the component normal to the sphere, leaving a vector tangent to the sphere.

With these notions in mind, we’re ready to see how flow matching can be adapted to manifold-valued data. Riemannian score-based diffusion models have already been explored; see De Bortoli *et al.* (2022) <d-cite key="de2022riemannian"></d-cite>.

---

## How Geometric Flow Matching Works

**Geometric Flow Matching (GFM)** extends the flow-matching idea to Riemannian manifolds (manifolds equipped with a notion of distance/metric). The core challenge is: **how do we define and train a flow entirely on a manifold?** We need a time-dependent vector field $$v(t,x)\in T_xM$$ and we need to match it to an “ideal” field that transports probability along a chosen path $$p_t$$ on $$M$$.

The recipe:

1. **Choose a path $$p_t$$ on the manifold.**  
   We still start with a simple distribution $$p_0$$ on $$M$$ and the data distribution $$p_1$$. One convenient choice is a *geodesic interpolation* between a data sample $$x_1\sim p_1$$ and a random sample $$x_0\sim p_0$$:

   $$
   \begin{align*}
   \psi_t(x_0 \mid x_1) = \exp_{x_0}\left( t \cdot \log_{x_0}(x_1) \right)
   \end{align*}
   $$

   At $$t=0$$, $$\psi_0 = x_0$$; at $$t=1$$, $$\psi_1 = x_1$$ <d-cite key="chen2023flow"></d-cite>.

2. **Define the target velocity field.**  
   Differentiate the interpolation:

   $$
   \begin{align*}
   u(t, x_t) = \partial_t \psi_t(x_0 \mid x_1) \bigg|_{x_t = \psi_t} \in T_{x_t}M
   \end{align*}
   $$

   For a geodesic path, the target velocity has a closed form $$u(t,x_t)=\frac{\log_{x_t}(x_1)}{1-t}, \qquad 0\le t<1$$. It is simply the “remaining” log-vector divided by the residual time $$(1-t)$$.
   <!-- For geodesics, $$u$$ is the remaining log-vector to $$x_1$$ scaled by $$1/(1-t)$$. -->

3. **Train a neural network to match $$u$$.**  
   Sample $$(t,x_0,x_1)$$, compute $$x_t = \psi_t$$, and minimize:

   $$
   \begin{align*}
   \mathcal{L} = \mathbb{E}\left[ \left\| v_\theta(t, x_t) - u(t, x_t) \right\|^2 \right]
   \end{align*}
   $$

Integrating the learned ODE:

$$
\begin{align*}
\dot{x} = v_\theta(t, x)
\end{align*}
$$

from $$t=0$$ to $$1$$ then carries $$p_0$$ to $$p_1$$, and **the state $$x(t)$$ stays on $$M$$ by construction**.

---

### Additional Intuition and Insights

Under the hood, several things ensure this is mathematically consistent:

$$
\begin{align*}
\dot{x}(t) = v(t, x(t)),\quad &x(t) \in M,\quad v(t,x) \in T_x M \\
\Rightarrow\ & \text{flow remains on } M,\ \text{ideally matching } p_t \text{ if } v = u
\end{align*}
$$

Chen & Lipman’s RFM work <d-cite key="chen2023flow"></d-cite> showed that GFM avoids:
- simulating stochastic processes on $$M$$,
- computing divergence of $$v$$ on the manifold,
- and enables closed-form $$u(t,x)$$ in many geometries.

These advantages come at no extra simulation cost. For manifolds where geodesics aren’t explicit, spectral methods can approximate $$\exp$$ and $$\log$$ as needed. 

Let’s build some intuition with a simple manifold: the **circle** ($$S^1$$). Suppose $$p_0$$ is uniform noise on the circle, and $$p_1$$ concentrates around a few angles. If $$x_0 \sim p_0$$ and $$x_1 \sim p_1$$, the geodesic between them is a short arc. The geodesic interpolation:

$$
\begin{align*}
\psi_t(x_0 \mid x_1) = \exp_{x_0}\left( t \cdot \log_{x_0}(x_1) \right)
\end{align*}
$$

describes a smooth movement along that arc. Then the vector field $$u(t,x)$$ points along the circle toward data-rich regions — a kind of breeze pushing probability mass toward $$p_1$$. Crucially, **all motion stays on the manifold**, unlike linear paths in $$\mathbb{R}^2$$.

For general manifolds, the learned vector field is also tangent:

$$
\begin{align*}
\dot{x}(t) = v_\theta(t, x(t)) \in T_{x(t)}M
\end{align*}
$$

The exp map acts internally in integration: you take a small step in the tangent space, then reproject:

$$
\begin{align*}
x_{t+\epsilon} = \exp_{x_t}(\epsilon \cdot v_\theta(t,x_t))
\end{align*}
$$

This can be done via Lie group tools on $$\mathrm{SO}(3)$$ or $$\mathrm{SE}(3)$$ using axis-angle representation.

Beyond using a fixed metric, a related extension—**Metric Flow Matching (MFM)**—first *learns* a data-driven Riemannian metric and then applies FM inside that learned geometry <d-cite key="kapusniak2024metric"></d-cite>.

---

### Symmetry and Equivariance

On many manifolds (like $$\mathrm{SE}(3)$$), **invariance** matters. If you rotate a protein and then generate, it should be the same as generating and then rotating.

$$
\begin{align*}
G(x) \sim p_1 \quad \Leftrightarrow \quad x \sim p_1,\quad \text{for all } G \in \mathrm{SE}(3)
\end{align*}
$$

This is called **equivariance**, and GFM can preserve it naturally by choosing symmetric $$p_0$$ and $$p_1$$. For example, FoldFlow <d-cite key="bosese"></d-cite> uses:
- $$p_0$$: SE(3)-invariant prior (random rigid motions of a base scaffold)
- $$p_1$$: data distribution of protein backbones (also SE(3)-invariant)

Then, GFM learns relative transformations without fixing any global frame:

$$
\begin{align*}
\text{Learned flow:} \quad p_0^{\text{inv}} \longrightarrow p_1^{\text{inv}} \text{ via SE(3)-equivariant vector field}
\end{align*}
$$

This avoids accidental alignment artifacts.

---

### Summary

Geometric Flow Matching adapts FM using exp/log and geodesics to define a time-dependent velocity field on $$M$$. The network learns to project samples toward $$p_1$$ by "blowing" them along the manifold.

$$
\begin{align*}
\psi_t(x_0 \mid x_1) &= \exp_{x_0}(t\,\log_{x_0}(x_1)) \\
v_\theta(t,x) &\approx \partial_t \psi_t(x_0 \mid x_1) \\
\mathcal{L} &= \mathbb{E}_{t, x_0, x_1}\left\| v_\theta(t, \psi_t) - \partial_t \psi_t \right\|^2
\end{align*}
$$

Even with these formulas, the core message is:

$$
\begin{align*}
\text{Move points along geodesics toward high-density regions on } M
\end{align*}
$$

rather than interpolating straight in Euclidean space. That’s the essence of GFM.

---

## Case Study - Protein Modeling with GFM

<!-- Below we highlight representative works that leverage GFM for proteins.	For context, SE(3)-equivariant diffusion models remain a strong baseline in this area <d-cite key="yim2023se"></d-cite>. -->

### Why is GFM a Natural Fit for Proteins?

Protein structure generation sits at the intersection of **stringent geometric constraints** and **high-dimensional variability**:

1. **Hierarchical manifolds.**  
   * **Backbone pose** lives in the special Euclidean group $$\mathrm{SE}(3)$$.  
   * **Backbone torsion angles** live on $S^1$ circles.  
   * **Side-chain rotamers** occupy a high-dimensional torus $$\mathbb{T}^n$$.  
   GFM operates directly on these product manifolds, guaranteeing every intermediate and final sample obeys the physics **by construction**, rather than relying on post-hoc projection or rejection.

2. **Geodesic realism.**  
   Small, local changes in a protein’s conformation correspond to *geodesic* moves on these manifolds (e.g. screw motions in $$\mathrm{SE}(3)$$ or incremental bond rotations). GFM’s flows follow geodesics, so probability mass moves along physically plausible trajectories that mirror molecular dynamics.

3. **Built-in symmetries.**  
   Proteins have no preferred global frame—rotating or translating an entire molecule should not change likelihood. Because GFM learns *tangent* vector fields on the manifold, enforcing **SE(3)-equivariance** is straightforward, eliminating spurious frame-dependence.

4. **Deterministic and differentiable trajectories.**  
   Unlike diffusion models that require hundreds of noisy steps, a trained GFM ODE can often transform noise to structure in **tens of steps**. The deterministic path is differentiable end-to-end, enabling gradient-based design or refinement tasks.

5. **Sample validity and efficiency.**  
   Empirically, GFM yields near-100 % valid backbones and side-chains, while Euclidean flows or diffusion models need costly relaxation to fix bond-length or chirality violations. Matching velocities (not log-densities) also removes expensive divergence terms, giving **1–2× faster training** on large structure sets.

---

### Representative Works

* **FoldFlow (ICLR 2024)** – deterministic / stochastic flows on $$\mathrm{SE}(3)$$ for protein backbones <d-cite key="bosese"></d-cite>. Stable, faster than diffusion; up to 300 residues.

$$
\begin{align*}
\text{Manifold:} \quad \mathrm{SE}(3) \quad \text{(rotation + translation)} \\
\text{Application:} \quad \text{protein backbone generation}
\end{align*}
$$

* **FlowPacker (Bioinformatics 2025)** – torsional flow matching for side-chain packing on a high-dimensional torus; respects $$360^\circ/180^\circ$$ symmetries; beats diffusion baselines <d-cite key="lee2025flowpacker"></d-cite>.

$$
\begin{align*}
\text{Manifold:} \quad \mathbb{T}^n \quad \text{(n-dimensional torus)} \\
\text{Property:} \quad \text{periodicity of angles (e.g., } 360^\circ \sim 0^\circ \text{)}
\end{align*}
$$

* **Proteína (ICLR 2025)** – massive flow model (21M structures) generating up to 800-residue proteins with hierarchical conditioning <d-cite key="geffnerproteina"></d-cite>.

$$
\begin{align*}
\text{Scale:} \quad \sim 21\text{M proteins},\quad \leq 800 \text{ residues}
\end{align*}
$$

* **Pullback Flow Matching (NeurIPS 2024)** – learns an isometric latent manifold first, then runs FM <d-cite key="de2024pullback"></d-cite>.

$$
\begin{align*}
\text{Approach:} \quad \text{latent manifold learning} \\
\text{Technique:} \quad \text{isometric pullback + flow matching}
\end{align*}
$$

These works show GFM’s improved validity, speed, and scalability over manifold-naïve methods.

---

## Conclusion

Geometric Flow Matching (GFM) shows that weaving rigorous differential geometry into deep generative modeling unlocks a new level of fidelity, efficiency, and scientific utility when the data itself lives on curved spaces. For proteins, a GFM‑trained flow respects bond geometry, SE(3) symmetries, and torsional periodicities at every timestep, outputting structures that are ready for downstream design or simulation with almost no post‑processing.

But proteins are only the beginning. Camera trajectories in robotics and AR/VR, global climate fields on Earth’s sphere, crystalline orientations in materials science, and articulated body poses in computer graphics all inhabit non‑Euclidean manifolds. Early adopters already report faster convergence, higher validity, and richer sample diversity than Euclidean flows or diffusion baselines.

The success of FoldFlow, FlowPacker, Proteína, and Pullback FM crystallizes a simple lesson: geometry is not a decorative extra—it is the very substrate on which real‑world data lives. GFM offers a blueprint for embracing that curvature without sacrificing the scalability of modern deep learning. If history is any guide, the next wave of breakthroughs—from novel enzymes to autonomous drones—will be powered by models that are as geometrically faithful as the phenomena they seek to capture.

---
