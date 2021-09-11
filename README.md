# trackun

A Python package for (multiple) object tracking using recursive Bayesian filtering

## **Demo**

**Example 1**: GM-Bernoulli filter for single-object tracking on Constant Velocity with Gaussian noise model

```bash
python demo.py -s -m linear_gaussian -f GM-Bernoulli -o vis/output
```

<details>
  <summary>Expand</summary>

![Examples of GM-Bernoulli filter](images/gm-bernoulli.gif)

</details>

**Example 2**: GM-PHD and GM-CPHD filter for multi-object tracking on Constant Velocity with Gaussian noise model

```bash
python demo.py -m linear_gaussian -f GM-CPHD GM-PHD -o vis/output
```

<details>
  <summary>Expand</summary>

![Examples of GM-PHD and GM-CPHD filter](images/gm-cphd-phd.gif)

</details>

## **Checklist**

<details>
  <summary>Click to expand!</summary>

### Filters

<details>
  <summary>Click to expand!</summary>

- [ ] Single Object
  - [ ] Kalman Filter (GMS)
  - [ ] Particle Filter (SMC)
  - [ ] Extended Kalman Filter (EKF)
  - [ ] Unscented Kalman Filter (UKF)
- [ ] Bernoulli
  - [x] Kalman Filter (GMS)
  - [ ] Particle Filter (SMC)
  - [ ] Extended Kalman Filter (EKF)
  - [ ] Unscented Kalman Filter (UKF)
- [ ] Probability Hypothesis Density (PHD)
  - [x] Kalman Filter (GMS)
  - [ ] Particle Filter (SMC)
  - [ ] Extended Kalman Filter (EKF)
  - [ ] Unscented Kalman Filter (UKF)
- [ ] Cardinalized Probability Hypothesis Density (CPHD)
  - [x] Kalman Filter (GMS)
  - [ ] Particle Filter (SMC)
  - [ ] Extended Kalman Filter (EKF)
  - [ ] Unscented Kalman Filter (UKF)
- [ ] Robust Probability Hypothesis Density (PHD)
  - [ ] Unknown clutter (Lambda-CPHD)
    - [ ] Kalman Filter (GMS)
    - [ ] Particle Filter (SMC)
    - [ ] Extended Kalman Filter (EKF)
    - [ ] Unscented Kalman Filter (UKF)
  - [ ] Unknown detection probability (pD-CPHD)
    - [ ] Kalman Filter (GMS)
    - [ ] Particle Filter (SMC)
    - [ ] Extended Kalman Filter (EKF)
    - [ ] Unscented Kalman Filter (UKF)
  - [ ] Unknown clutter rate and detection probability
    - [ ] Kalman Filter (GMS)
    - [ ] Particle Filter (SMC)
    - [ ] Extended Kalman Filter (EKF)
    - [ ] Unscented Kalman Filter (UKF)
- [ ] Cardinality Balanced Multi-target Multi-Bernoulli (CBMeMBer)
  - [ ] Kalman Filter (GMS)
  - [ ] Particle Filter (SMC)
  - [ ] Extended Kalman Filter (EKF)
  - [ ] Unscented Kalman Filter (UKF)
- [ ] Generalized Labeled Multi-Bernoulli (GLMB)
  - [ ] Kalman Filter (GMS)
  - [ ] Particle Filter (SMC)
  - [ ] Extended Kalman Filter (EKF)
  - [ ] Unscented Kalman Filter (UKF)
- [ ] Labeled Multi-Bernoulli (LMB)
  - [ ] Kalman Filter (GMS)
  - [ ] Particle Filter (SMC)
  - [ ] Extended Kalman Filter (EKF)
  - [ ] Unscented Kalman Filter (UKF)
  
</details>

### **Models**

<details>
  <summary>Click to expand!</summary>

#### Motion model

- [ ] Linear
  - [x] Constant velocity
- [ ] Non-Linear
  - [ ] Coordinated turn (CT)
- [ ] General (?)

#### Measurement model

- [x] Linear
- [ ] Non-Linear/Gen
  - [ ] Bearing
- [ ] General (?)

#### Other models

- [ ] Birth model
  - [x] Multi-Bernoulli Gaussian
  - [x] Multi-Bernoulli Gaussian Mixture
- [ ] Detection model
  - [x] Constant detection probability
- [ ] Survival model
  - [x] Constant survival probability
- [ ] Clutter model
  - [x] Uniform clutter
  
</details>

### Metrics

- [x] OSPA
- [ ] OSPA2

### Utility

- [ ] Examples and Visualization
- [ ] Benchmarking
- [ ] Optimization (consider memory-speed tradeoffs, JIT,...)
- [ ] System design and folder structure
- [ ] Testing

</details>

## **Credits**

Original MATLAB implementation comes from http://ba-tuong.vo-au.com/codes.html
