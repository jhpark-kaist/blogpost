---
layout: distill
title: AI810 Blog Post (20248084)
description: Generative models traditionally operate in Euclidean space. However, many types of data are naturally manifold-valued, meaning they live on curved or constrained spaces rather than a flat plane. In this article, we explore Geometric Flow Matching (GFM an emerging framework that adapts flow-based generative modeling to work natively on manifolds. We will build intuition for why this is needed (especially for protein structures), review how standard Flow Matching works in Euclidean settings, and then explain the manifold generalization. Along the way, we use analogies, minimal math, and visual aids to keep things intuitive yet rigorous.
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

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
# toc:
#   - name: Viewing Images in Frequency Domain
#   - name: Analysis of Bias in GANs
#   # you can additionally add subentries like so
#     subsections:
#     - name: Setting Up the Generative CNN Structure
#     - name: ReLU as a Fixed Binary Mask
#     - name: Onto The Analysis of Filter Spectrum
#   - name: Frequency bias in Diffusion Models
#   - name: Mitigation of Frequency Bias Using Spectral Diffusion Model
#   - name: Conclusion
---

## 1. Why Generative Models on Manifolds?

Real-world data often have inherent geometric constraints that make their natural domain a manifold instead of a full Euclidean space. **Proteins** provide an excellent example: a protein’s 3-D conformation can be described by torsion angles (like phi/psi backbone angles or side-chain rotations), which are periodic variables on a circle ($S^1$). Similarly, the overall orientation and position of a protein in space lie in the $$\mathrm{SE}(3)$$ group of 3-D rotations and translations – another manifold. If we naively treat such data as unconstrained vectors in $$\,\mathbb{R}^n\,$$, we risk modeling artifacts:

* **Periodic Angles** – Imagine modeling a dihedral angle that wraps around at $$360^\circ$$. In Euclidean representation, $$0^\circ$$ and $$360^\circ$$ appear far apart, even though they are physically the same conformation. A Euclidean generative model might struggle, placing undue probability mass at the artificial boundaries (e.g. around $$0^\circ$$ vs $$360^\circ$$) because it doesn’t “know” the variable is circular.  
* **Rotations** – A protein domain’s orientation is a point on the sphere of rotations (technically $$\mathrm{SO}(3)$$). If we parameterize orientation by Euler angles in $$\,\mathbb{R}^3\,$$, the model can suffer from discontinuities (gimbal lock, etc.) and non-uniform coverage of orientation space. For instance, sampling each Euler angle from a Gaussian does not yield a uniform random rotation – the distribution gets warped by the coordinate parameterization.  
* **Rigid Motions** – When placing multiple protein subunits relative to each other, their relative pose is in $\mathrm{SE}(3)$. Treating this as six independent numbers (three for translation, three for rotation) in $\,\mathbb{R}^6\,$ ignores the fact that rotations have a curved geometry. A naive model might generate “average” rotation matrices that are not valid rotations, or concentrate samples in certain regions of orientation space.  

These issues hint that to model such manifold-valued data, we should respect the geometry. Indeed, machine-learning researchers have noted that many observations in vision, graphics, and biology are non-Euclidean, and purely Euclidean generative techniques are inadequate <d-cite key="zhen2021flow"></d-cite>. For proteins, recent successes in generative modeling (like diffusion models for protein structures) had to incorporate geometric awareness. For example, the **Proteus** diffusion model built in triangle geometry to generate protein backbones without a pre-trained guide <d-cite key="wang2024proteus"></d-cite>. In general, a generative model “aware” of the manifold can sample more valid and realistic data by avoiding distortions that arise from flattening the manifold incorrectly.  

### Why not just embed the manifold in $\mathbb{R}^n$?  

One might think we can bypass the issue by representing manifold data in some Euclidean coordinates (for example, using 3-D Cartesian coordinates for points on a sphere). However, while you can represent any manifold data in $$\mathbb{R}^n$$, the distribution often becomes complicated in those coordinates. It might concentrate on a lower-dimensional surface in $$\mathbb{R}^n$$ or have discontinuous support. Generating new samples in the ambient space will likely produce points off the manifold (e.g. points not exactly on the sphere or violating constraints). Enforcing the manifold constraints after generation can be non-trivial or can introduce bias. Hence, there is strong motivation to build generative models that operate directly on the manifold, ensuring all samples and intermediate states respect the underlying geometric constraints.

Protein modeling epitomizes this need. A protein backbone is essentially a series of bond angles and lengths – many of which (bond lengths, planar angles) are fixed, leaving torsion angles that rotate freely but periodically. Traditional generative models that ignore these periodicities might output physically implausible structures or require ad-hoc fixes (like wrapping angles or rejecting invalid outputs). By developing generative models on manifolds, researchers aim to produce proteins natively in valid conformation spaces (like torsion-angle space or $$\mathrm{SE}(3)$$ space) without falling off the manifold. The same principle applies to other domains (robotics with rotation trajectories, climate data on Earth’s sphere, etc.), but we’ll keep our focus on proteins as a running example.

---

## 2. Recap: Flow Matching in Euclidean Space

Before diving into manifold techniques, let’s briefly review **Flow Matching (FM)** in the standard Euclidean setting. Flow Matching is a relatively new paradigm for training *continuous normalizing flows* (CNFs) – a class of generative models where you transform a simple distribution (like Gaussian noise) into the data distribution by integrating an ODE (ordinary differential equation). Traditionally, CNFs are trained by maximum likelihood, which involves computing Jacobian determinants or simulating differential equations, which can be expensive and unstable. Flow Matching takes a different route: **don’t simulate the entire flow during training – instead, directly match the velocity field** <d-cite key="lipman2022flow"></d-cite>.

In Flow Matching, one first defines a smooth path of probability distributions $p_t$ for $$t\!\in\![0,1]$$ that connects the known source distribution $$p_{t=0}$$ (usually something simple like $$\mathcal{N}(0,I)$$ noise) to the target data distribution $$p_{t=1}$$ (the distribution of your dataset). Think of this path as a plan for how a blob of probability mass should morph from noise into data. A common choice is a Gaussian diffusion path (like gradually adding noise to data or vice versa), but other choices are possible (e.g. an optimal-transport interpolation). Flow Matching doesn’t commit to a specific simulation process; it just needs a prescribed $p_t$.

The goal is to find a time-dependent vector field $$v(t,x)$$ such that if we move particles according to the ODE $$\frac{d\mathbf{x}}{dt}=v(t,\mathbf{x})$$, it will carry out the transformation from $p_0$ to $p_1$ following the path $p_t$. In other words, $v(t,x)$ should be the ideal velocity field that transports the distribution along the chosen path. If we knew this ideal $v$, we could generate samples by integrating $$\mathbf{x}(0)\!\sim\!p_0$$ through the ODE to $$\mathbf{x}(1)\!\sim\!p_1$$.

Flow Matching gives a clever way to train a neural network to approximate $$v(t,x)$$ **without ever simulating the ODE during training**. Instead, one samples pairs $$(t,x_t,x_1)$$ where $$x_t\!\sim\!p_t$$ and $$x_1\!\sim\!p_1$$, and then the network is asked to predict the “target” velocity $$u(t,x_t)$$ that would point $x_t$ toward a sample from the later distribution. In practice, one can derive an analytical form for this target velocity under many paths. For example, with a diffusion (Gaussian) interpolation, the target vector field is proportional to the *score* (gradient of log-density) or the difference between $$x_t$$ and $$x_1$$ in certain cases 4. The training objective is a regression loss: make $$v_\theta(t,x_t)\approx u(t,x_t)$$ in mean-squared error. This way, the model “matches” the flow velocities, hence the name. Crucially, this procedure is simulation-free – we don’t have to integrate the ODE in inner loops; we only need to sample from $$p_t$$ and $$p_1$$, which is often easy by construction.

Flow Matching has notable advantages. By avoiding back-propagation through ODE solvers or computing log-determinants, it **scales to large problems and tends to be more stable**. In fact, using diffusion paths, Flow Matching was shown to train diffusion models more robustly than the usual score-matching approach 5 6. It also allows exploring non-diffusive paths: Lipman *et al.* demonstrated that using an optimal-transport (OT) path (which linearly interpolates distributions in a mass-preserving way) yields faster convergence and better samples than standard diffusion 6. In summary, in Euclidean space FM provides a flexible framework to train generative flows by “teaching” a vector field how to morph one distribution into another, rather than simulating it blindly.

**Limitations:** Flow Matching as described assumes we can freely sample and compute differences in the ambient space $$\mathbb{R}^n$$. If our data lies on a manifold, applying Euclidean FM directly can be problematic. For instance, if $$x_t$$ and $$x_1$$ are points on a sphere or in $$\mathrm{SO}(3)$$, subtracting them or mixing them linearly is not geometrically meaningful (you can’t just average two rotations by component-wise linear interpolation without leaving $$\mathrm{SO}(3)$$). We need to reformulate the idea of a “conditional path” and “velocity field” in a way that lives on the manifold. **This is where Geometric Flow Matching enters the scene.**

---

## 3. Preliminaries: Manifolds

Before tackling flow matching on manifolds, let’s ensure we understand some basics of manifold geometry.

A **manifold** is a space that may be curved or constrained, but locally it looks like a flat Euclidean space. Classic examples include the surface of a sphere, a torus (doughnut shape), or the rotation group $$\mathrm{SO}(3)$$ (which topologically is like a 3-D sphere with opposite points identified). Formally, a $$d$$-dimensional manifold $$M$$ is something that can be covered by coordinate patches (called *charts*), each of which is like a mapping from an open set of $$\,\mathbb{R}^d$$ to $$M$$. However, no single flat chart can cover the entire manifold without distortion or singularities, much like no flat map can perfectly represent the entire Earth.

**Analogy (Earth Maps):** Think of the Earth (a sphere) as our manifold. We can define latitude and longitude as coordinates – that’s a chart covering most of the globe. But it breaks down at the poles (longitude is undefined at exactly the North Pole). We might use another chart for polar regions. Each chart is like a different “map projection” – Mercator, polar projection, etc. They all overlap partially and relate via coordinate transformations. So, working on a manifold often means you either restrict yourself to a chart (and remember to switch charts when needed), or you do calculations in a coordinate-free way using geometric constructs.

One key construct is the **tangent space**. At any point $p$ on the manifold, the tangent space $$T_pM$$ is a flat $d$-dimensional space consisting of the possible “directions” one can move infinitesimally from $p$ on the manifold. If you’ve seen a tangent line to a curve or a tangent plane to a surface, that’s the idea – it “just touches” the manifold at that point and is the best linear approximation of the manifold near $$p$$. For a sphere, the tangent space at a point is the plane tangent to the sphere at that point.

> A sphere with a tangent plane at point P. The plane represents the tangent space $$T_PM$$, a local linear approximation of the manifold. You can move in any direction along the tangent plane, which corresponds to moving along the sphere’s surface (locally) in some direction.

Why do we care about tangent spaces? Because calculus and differential equations on manifolds are formulated in terms of tangent vectors. A velocity or direction of motion at a point on the manifold lives in the tangent space. If we want to define a vector field $$v(t,x)$$ on a manifold, we must ensure $$v(t,x)\in T_xM$$ (it points along the manifold, not poking out of it).

Another essential pair of tools are the **exponential map** and **logarithmic map** (exp and log for short). These maps help us go back-and-forth between the manifold and tangent spaces in a well-defined way:

* **Log map $$\log_p(q)$$** – Given two points $$p,q\in M$$, the log map at $$p$$ yields a tangent vector $$v\in T_pM$$ which points from $$p$$ in the direction of $$q$$, with a length equal to the geodesic distance from $$p$$ to $$q$$. In intuitive terms, $$\log_p(q)$$ answers: “if I start at $p$ and walk toward $$q$$ along the shortest path on the manifold, what direction and distance should I go initially?”  
* **Exp map $$\exp_p(v)$$** – This is (locally) the inverse of log. Given a tangent vector $$v\in T_pM$$, $$\exp_p(v)$$ gives the point $$q$$ on the manifold you reach by starting at $$p$$ and moving in the direction of $$v$$ for a unit time (or until you cover the distance $$|v|$$).  

On a Euclidean plane, $$\exp_p(v)=p+v$$ and $$\log_p(q)=q-p$$. But on a sphere, these operations involve trigonometry (great-circle arcs). On $$\mathrm{SO}(3)$$, $$\exp$$ and $$\log$$ can be related to matrix exponentials and rotation vectors.

The exp/log maps give a way to translate problems on a manifold into problems in a tangent (Euclidean) space, do some computation, and perhaps map back. For instance, to interpolate between two manifold points $$p$$ and $$q$$, a natural path is the **geodesic**: $$\gamma(t)=\exp_p\!\bigl(t\,\log_p(q)\bigr)$$ for $$0\le t\le1$$. This yields a curve that starts at $$p$$ ($$t=0$$) and ends at $$q$$ ($$t=1$$).

Finally, if we do embed the manifold in a higher-dimensional Euclidean space (like a sphere in $$\mathbb{R}^3$$), we can talk about **projecting** vectors onto the tangent space. For example, if you have some ambient vector in $$\mathbb{R}^3$$ at point $$p$$ on the sphere, the tangent projection would subtract the component normal to the sphere, leaving a vector tangent to the sphere.

With these notions in mind, we’re ready to see how flow matching can be adapted to manifold-valued data.

---

## 4. How Geometric Flow Matching Works

**Geometric Flow Matching (GFM)** extends the flow-matching idea to Riemannian manifolds (manifolds equipped with a notion of distance/metric). The core challenge is: **how do we define and train a flow entirely on a manifold?** We need a time-dependent vector field $$v(t,x)\in T_xM$$ and we need to match it to an “ideal” field that transports probability along a chosen path $$p_t$$ on $$M$$.

The recipe:

1. **Choose a path $$p_t$$ on the manifold.**  
   We still start with a simple distribution $$p_0$$ on $$M$$ and the data distribution $p_1$. One convenient choice (Chen & Lipman 2023) is a *geodesic interpolation* between a data sample $$x_1\sim p_1$$ and a random sample $$x_0\sim p_0$$:  
   $$\displaystyle \psi_t(x_0\mid x_1)=\exp_{x_0}\!\bigl(t\,\log_{x_0}(x_1)\bigr).$$  
   At $$t=0$$, $$\psi_0=x_0$$; at $$t=1$$, $$\psi_1=x_1$$.

2. **Define the target velocity field.**  
   Differentiate the interpolation:  
   $$\,u(t,x_t)=\partial_t\psi_t(x_0\mid x_1)\,\bigl|_{x_t=\psi_t}\in T_{x_t}M.$$  
   For geodesics, $u$ is essentially the remaining log-vector to $$x_1$$ scaled by $$1/(1-t)$$.

3. **Train a neural network to match $u$.**  
   Sample $(t,x_0,x_1)$, compute $x_t=\psi_t$, and minimize  
   $\mathcal{L}=\mathbb{E}\,\bigl\|v_\theta(t,x_t)-u(t,x_t)\bigr\|^2$.  

Integrating the learned ODE $\dot{x}=v_\theta(t,x)$ from $t=0$ to $1$ then carries $p_0$ to $p_1$, and **the state $x(t)$ stays on $M$ by construction**.

A circle-example intuition: Points are “blown” along arcs toward high-density regions, never leaving the circle, unlike straight-line interpolation through its interior.

---

## 5. Case Study: Protein Modeling with GFM

* **FoldFlow (ICLR 2024)** – deterministic / stochastic flows on $\mathrm{SE}(3)$ for protein backbones 10 11 12. Stable, faster than diffusion; up to 300 residues.  
* **FlowPacker (NeurIPS MLSB 2024)** – torsional flow matching for side-chain packing on a high-dimensional torus; respects $360^\circ/180^\circ$ symmetries; beats diffusion baselines 13 14.  
* **Proteína (ICLR 2025)** – massive flow model (21 M structures) generating up to 800-residue proteins with hierarchical conditioning 15 16 17 18.  
* **Pullback Flow Matching (NeurIPS 2024)** – learns an isometric latent manifold first, then runs FM 19 20 27.

These works show GFM’s improved validity, speed, and scalability over manifold-naïve methods.

---

## 6. Strengths, Practical Considerations, and Challenges

### Strengths of GFM  

* **Geometry fidelity** – samples and trajectories never leave $M$.  
* **Stability & efficiency** – no inner-loop ODE simulation; fewer steps at inference 22.  
* **Flexible paths** – geodesic or OT paths simplify learning 6 9.  
* **Equivariance & conditioning** – combines naturally with SE(3)-equivariant nets, classifier-free or RL guidance 23 24 25 26.  
* **Likelihoods** – invertible flows allow density evaluation when required.

### Implementation considerations  

* Provide $\exp/\log$ formulas (sphere, $\mathrm{SO}(3)$, $\mathrm{SE}(3)$, torus).  
* Output vectors then project onto $T_xM$.  
* Pick a convenient base distribution $p_0$.  
* Use equivariant graph networks / transformers for $v_\theta$.  
* Handle numerical stability near singularities (e.g. antipodal rotations).  
* For exotic $M$, spectral methods approximate geodesics 9.

### Current challenges & research directions  

* Very high-dimensional manifolds (long proteins).  
* Unknown manifolds → latent-space learning (PFM).  
* Hybrid FM + EBM models 28.  
* Theoretical guarantees on non-compact $M$.  
* User-specified design constraints (e.g. guided protein design, FoldFlow-2’s ReFT) 29.  
* Broader tooling and educational resources.

---

## Conclusion

Geometric Flow Matching marries differential geometry with deep generative modeling. By respecting curvature instead of fighting it, GFM yields accurate, valid, and efficient models – already transforming protein design and poised to impact many other areas. The momentum of FoldFlow, FlowPacker, Proteína, and Pullback FM signals that geometry-aware AI is at the cutting edge, with much more ahead.

---

## References

1 [2012.10013] *Flow-based Generative Models for Learning Manifold-to-Manifold Mappings*  
2 3 *Proteus: Exploring Protein Structure Generation for Enhanced Designability and Efficiency*  
4 5 6 22 [2210.02747] *Flow Matching for Generative Modeling*  
7 8 9 [2302.03660] *Flow Matching on General Geometries*  
10 11 12 [2310.02391] *SE(3)-Stochastic Flow Matching for Protein Backbone Generation*  
13 14 *FlowPacker: Protein Side-Chain Packing with Torsional Flow Matching*  
15 16 17 18 25 26 *Proteína: Scaling Flow-based Protein Structure Generative Models*  
19 20 27 [2410.04543] *Pullback Flow Matching on Data Manifolds*  
21 *Fisher Flow Matching for Generative Modeling over Discrete Data*  
23 24 29 [2405.20313] *Sequence-Augmented SE(3)-Flow Matching for Conditional Protein Backbone Generation*  
28 *Unifying Flow Matching and Energy-Based Models for Generative …*  
