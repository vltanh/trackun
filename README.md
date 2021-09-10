# trackun

A Python package for (multiple) object tracking

# Checklist

## Filters

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

## Models

### Motion model

- [x] Linear
- [ ] Coordinated turn (CT)

### Observation model

- [x] Linear
- [ ] Bearing

## Metrics

- [x] OSPA
- [ ] OSPA2

## Utility

- [ ] Examples and Visualization
- [ ] Benchmarking
- [ ] Optimization
- [ ] System design and folder structure

# Demo

![Examples of PHD and CPHD filter](visualize/gms_cphd_phd.gif)

# Credits

Original MATLAB implementation comes from http://ba-tuong.vo-au.com/codes.html
