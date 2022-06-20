extern crate nalgebra as na;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand_distr::StandardNormal;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use na::{Complex, ComplexField};

pub fn hilbert_schmidt_distance(sigma: &na::Matrix4<Complex<f64>>, rho: &na::Matrix4<Complex<f64>>) -> f64 {
    ( (sigma-rho).adjoint() * (sigma-rho) ).trace().abs()
}

pub fn trace_distance_herm(sigma: &na::Matrix4<Complex<f64>>, rho: &na::Matrix4<Complex<f64>>) -> f64 {
    let phi = sigma-rho;
    let phi_sqr = phi.adjoint() * phi;
    if (phi_sqr).trace() == Complex::new(0.0, 0.0) {
        return 0.0;
    }
    let eigs = (phi_sqr).eigenvalues().unwrap();
    let eigs_sqrt = eigs.map(|x| ComplexField::abs(x).sqrt());
    eigs_sqrt.sum()
}

pub fn random_pos_sphere(rng: &mut ThreadRng) -> (f64, f64) {
    let phi: f64 = Uniform::new_inclusive(0.0, 2.0*PI).sample(rng);
    let costheta: f64 = Uniform::new_inclusive(-1.0, 1.0).sample(rng);
    let theta: f64 = f64::acos(costheta);
    (theta, phi)
}

pub fn ginibre(rng: &mut ThreadRng, n: usize) -> na::DMatrix<Complex<f64>> {
    let g_r = na::DMatrix::from_fn(n, n, |_,_| na::Complex::new(rng.sample(StandardNormal), 0.0));
    let g_i = na::DMatrix::from_fn(n, n, |_,_| na::Complex::new(0.0, rng.sample(StandardNormal)));
    let g = g_r + g_i;
    g
}

pub fn random_qubit_pure(rng: &mut ThreadRng) -> na::Matrix2<Complex<f64>> {
    let (theta,phi) = random_pos_sphere(rng);
    let psi = na::Matrix2x1::new( Complex::new(f64::cos(theta/2.0),0.0), Complex::new(0.0, phi).exp() * Complex::new(f64::sin(theta/2.0),0.0) );
    let rho =  psi * psi.adjoint();
    rho
}

pub fn random_qubit_mixed(rng: &mut ThreadRng) -> na::Matrix2<Complex<f64>> {
    let g_dyn = ginibre(rng, 2);
    let g: na::Matrix2<Complex<f64>> = g_dyn.fixed_slice::<2, 2>(0, 0).into();
    let rho = g * g.adjoint();
    rho / rho.trace()
}

pub fn random_qubit(rng: &mut ThreadRng) -> na::Matrix2<Complex<f64>> {
    let pure: bool = rng.gen();
    if pure {
        return random_qubit_pure(rng);
    }
    else {
        return random_qubit_mixed(rng);
    }
}

pub fn bipartite_zero_discord(rng: &mut ThreadRng) -> na::Matrix4<Complex<f64>> {
    let zero = na::Matrix2::new( na::Complex::new(1.0,0.0), na::Complex::new(0.0,0.0), na::Complex::new(0.0,0.0), na::Complex::new(0.0,0.0) );
    let one = na::Matrix2::new( na::Complex::new(0.0,0.0), na::Complex::new(0.0,0.0), na::Complex::new(0.0,0.0), na::Complex::new(1.0,0.0) );
    let rho1 =  random_qubit(rng);
    let rho2 =  random_qubit(rng);
    let p1 = Uniform::new_inclusive(0.0, 1.0).sample(rng);
    let p2 = 1.0-p1;
    let chi = ( zero.kronecker(&rho1).scale(p1) ) + ( one.kronecker(&rho2).scale(p2) );
    chi
}

pub fn geometric_discord(rho_ab: &na::Matrix4<Complex<f64>>, n_times: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut distances: Vec<f64> = Vec::new();
    for _i in 0..=n_times {
        let chi = bipartite_zero_discord(&mut rng);
        let distance = hilbert_schmidt_distance(rho_ab, &chi);
        distances.push(distance);
        if distance == 0.0 {
            break;
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distances.first().unwrap().clone()
}

pub fn trace_distance_discord(rho_ab: &na::Matrix4<Complex<f64>>, n_times: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut distances: Vec<f64> = Vec::new();
    for _i in 0..=n_times {
        let chi = bipartite_zero_discord(&mut rng);
        let distance = trace_distance_herm(rho_ab, &chi);
        distances.push(distance);
        if distance == 0.0 {
            break;
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    0.5 * distances.first().unwrap().clone()
}

pub fn trace_distance_discord_parallel(rho_ab: &na::Matrix4<Complex<f64>>, n_times: usize, n_threads: u8) -> f64 {
    let rho_ab_in = rho_ab.clone();
    let distances_vec: Vec<f64> = Vec::new();
    let distances_arc = Arc::new(Mutex::new(distances_vec));
    let mut handles = vec![];
    for _ in 0..n_threads {
        let rho_ab_clone = rho_ab_in.clone();
        let distances_ref = Arc::clone(&distances_arc);
        let handle = thread::spawn(move || {
            let mut rng = rand::thread_rng();
            loop {
                let chi = bipartite_zero_discord(&mut rng);
                let distance = trace_distance_herm(&rho_ab_clone, &chi);
                let mut distances = distances_ref.lock().unwrap();
                if distances.len() < n_times {
                    distances.push(distance);
                    if distance == 0.0 {break;}
                }
                else {break;}
                }
            });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
    }
    let mut distances = distances_arc.lock().unwrap();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    0.5 * distances.first().unwrap().clone()
}

pub fn werner_state(lmda: f64) -> na::Matrix4<Complex<f64>> {
    let eye4 = na::Matrix4::<Complex<f64>>::identity();
    let psi_minus = na::Matrix4::new(
        Complex::new(0.0,0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0,0.0),
        Complex::new(0.0,0.0), Complex::new(0.5, 0.0), Complex::new(-0.5,0.0), Complex::new(0.0,0.0),
        Complex::new(0.0,0.0), Complex::new(-0.5,0.0), Complex::new(0.5, 0.0), Complex::new(0.0,0.0),
        Complex::new(0.0,0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0,0.0),
        );
    let werner = psi_minus.scale(lmda) + eye4.scale( ( 1.0-lmda)/4.0 );
    werner
}
