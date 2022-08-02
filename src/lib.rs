extern crate nalgebra as na;
use almost;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand_distr::StandardNormal;
use std::f64::consts::PI;
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use na::{Complex, ComplexField};
use nalgebra::linalg::SymmetricEigen;


#[derive(Copy, Clone)]
pub enum EntanglementMeasure {
    RelativeEntropyOfEntanglement,
}

#[derive(Copy, Clone)]
pub enum DiscordMeasure {
    GometricDiscord,
    TraceDistanceDiscord,
    RelativeEntropyDiscord,
}

#[derive(Copy, Clone)]
pub enum TotalCorrMeasure {
    TotalCorrelations
}

#[derive(Copy, Clone)]
pub enum QCorrelation {
    Entanglement(EntanglementMeasure),
    Discord(DiscordMeasure),
    //Total(TotalCorrMeasure),
}

#[derive(Copy, Clone)]
pub enum DistanceMetric {
    HilbertSchmidtDistance,
    TraceDistance,
}


#[allow(non_snake_case)]
pub fn logm(A: &na::Matrix4<Complex<f64>>, base: f64) -> na::Matrix4<Complex<f64>> {
    /* Matrix log of 4x4 matrix A. Currently only works for diagonalizable matrices. */
    let eigd = SymmetricEigen::new(A.clone());
    let V = &eigd.eigenvectors.clone();
    let V_inv = V.try_inverse().unwrap();
    let A_prime = V_inv * A * V;
    let log_A_p = A_prime.map(|z| -> Complex<f64> {
        if almost::zero(z.abs()) {
            Complex{re:0.0,im:0.0}
        }
        else {
            z.log(base)
        }
    });
    V * log_A_p * V_inv
}

pub fn partial_transpose(rho: &na::Matrix4<Complex<f64>>) -> na::Matrix4<Complex<f64>> {
    /* Partial transpose w.r.t. subsystem B of a 4x4 matrix ρ_AB. */
    let mut rho_pt = na::Matrix4::<Complex<f64>>::zeros();
    let indexes = vec![(0,0), (0,2), (2,0), (2,2)];
    for idx in indexes {
        let rho_block = rho.slice(idx, (2,2));
        let mut outslice = rho_pt.slice_mut(idx, (2,2));
        outslice.copy_from(&rho_block.transpose());
    }
    rho_pt
}

pub fn peres_horodecki_separable(rho: &na::Matrix4<Complex<f64>>) -> bool {
    /* Check whether a 4x4 density matrix is separable using the Peres–Horodecki criterion */
    let tol = -(1e-12);
    let rho_pt = partial_transpose(rho);
    let eigs_rho_pt = rho_pt.eigenvalues().unwrap();
    for i in 0..4 {
        if eigs_rho_pt[i].real() < tol {
            return false
        }
    }
    true
}

pub fn entropy_vn(rho: &na::Matrix4<Complex<f64>>, base: f64) -> f64 {
    /* von Neumann Entropy of a 4x4 density matrix */
    let eigs = rho.eigenvalues().unwrap();
    let mut nz_eigs = Vec::<f64>::new();
    for z in eigs.into_iter() {
        if ! almost::zero(z.abs()) {
            nz_eigs.push(z.real());
        }
    }
    let mut vn_entropy: f64 = 0.0;
    for eig in &nz_eigs {
        if base == 2.0 {
            vn_entropy -= eig * eig.log2();
        }
        else {
            vn_entropy -= eig * eig.log(base);
        }
    }
    vn_entropy
}

pub fn entropy_vn_logm(rho: &na::Matrix4<Complex<f64>>) -> f64 {
    /* von Neumann entropy. Original definition using matrix log. */
    -1.0 * (rho * logm(&rho,2.0)).trace().real()
}

#[allow(non_snake_case)]
pub fn entropy_relative(rho: &na::Matrix4<Complex<f64>>, sigma: &na::Matrix4<Complex<f64>>, base: f64) -> f64 {
    /* Relative entropy between two 4x4 density matrices.
       Based on Nielsen & Chuang and getting help from QuTiP code. */
    let mut S = -1.0 * entropy_vn(rho, base);
    let eigs_rho = SymmetricEigen::new(rho.clone());
    let eigs_sgm = SymmetricEigen::new(sigma.clone());
    let rvals = eigs_rho.eigenvalues.clone();
    let rvecs = eigs_rho.eigenvectors.clone();
    let svals = eigs_sgm.eigenvalues.clone();
    let svecs = eigs_sgm.eigenvectors.clone();
    let mut P = na::Matrix4::<f64>::zeros();

    for i in 0..rvals.len() {
        for j in 0..svals.len() {
            let m_p_ij = rvecs.column(i).adjoint() * svecs.column(j) * svecs.column(j).adjoint() * rvecs.column(i);
            let mut P_ij = m_p_ij[(0,0)].real();
            if almost::zero(P_ij) {
                P_ij = 0.0;
            }
            else if almost::equal(P_ij, 1.0) {
                P_ij = 1.0;
            }
            P[(i,j)] = P_ij;
            if almost::zero(svals[j]) && !( almost::zero(rvals[i]) || almost::zero(P_ij) ) {
                return f64::INFINITY;
            }
            else if !almost::zero(svals[j]) {
                if base == 2.0 {
                    S -= rvals[i] * P_ij * (svals[j].log2());
                }
                else {
                    S -= rvals[i] * P_ij * (svals[j].log(base));
                }
            }
            else {
                continue;
            }
        }
    }
    S
}

pub fn entropy_relative_logm(rho: &na::Matrix4<Complex<f64>>, sigma: &na::Matrix4<Complex<f64>>, base: f64) -> f64 {
    /* Naive implementation of relative entropy. Work in progress. */
    let s = ( -1.0 * entropy_vn(&rho,base) ) - ( rho * logm(&sigma,base) ).trace().real();
    s
}

pub fn hilbert_schmidt_distance(rho: &na::Matrix4<Complex<f64>>, sigma: &na::Matrix4<Complex<f64>>) -> f64 {
    /* Calculate the Hilbert Schmidt distance between two 4x4 matrices */
    ( (rho-sigma).adjoint() * (rho-sigma) ).trace().abs()
}

pub fn trace_distance_herm(rho: &na::Matrix4<Complex<f64>>, sigma: &na::Matrix4<Complex<f64>>) -> f64 {
    /* Calculate the trace distance between two 4x4 Hermitian matrices */
    let phi = rho-sigma;
    let phi_sqr = phi.adjoint() * phi;
    if almost::zero((phi_sqr).trace().abs()) {
        return 0.0;
    }
    let eigs = (phi_sqr).eigenvalues().unwrap();
    let eigs_sqrt = eigs.map(|x| ComplexField::abs(x).sqrt());
    eigs_sqrt.sum()
}

pub fn random_pos_sphere(rng: &mut ThreadRng) -> (f64, f64) {
    /* Select a random point on a unit sphere */
    let phi: f64 = Uniform::new_inclusive(0.0, 2.0*PI).sample(rng);
    let costheta: f64 = Uniform::new_inclusive(-1.0, 1.0).sample(rng);
    let theta: f64 = f64::acos(costheta);
    (theta, phi)
}

pub fn random_ginibre(rng: &mut ThreadRng, n: usize) -> na::DMatrix<Complex<f64>> {
    /* Generate a random matrix of size (n,n) from the Ginibre ensemble */
    let dense: bool = rng.gen();
    if dense {
        return random_ginibre_dense(rng, n)
    }
    else {
        return random_ginibre_sparse(rng, n)
    }
}

pub fn random_ginibre_dense(rng: &mut ThreadRng, n: usize) -> na::DMatrix<Complex<f64>> {
    /* Generate a dense random matrix of size (n,n) from the Ginibre ensemble */
    let g_r = na::DMatrix::from_fn(n, n, |_,_| na::Complex::new(rng.sample(StandardNormal), 0.0));
    let g_i = na::DMatrix::from_fn(n, n, |_,_| na::Complex::new(0.0, rng.sample(StandardNormal)));
    let g = g_r + g_i;
    g
}

pub fn random_ginibre_sparse(rng: &mut ThreadRng, n: usize) -> na::DMatrix<Complex<f64>> {
    /* Generate a sparse random matrix of size (n,n) from the Ginibre ensemble */
    let mut g = na::DMatrix::<Complex<f64>>::zeros(n,n);
    for i in 0..n {
        for j in 0..n {
            let non_zero: bool = rng.gen();
            if non_zero {
                g[(i,j)] = na::Complex::new(rng.sample(StandardNormal), 0.0) + na::Complex::new(0.0, rng.sample(StandardNormal));
            }
            else {
                g[(i,j)] = na::Complex::new(0.0, 0.0);
            }
        }
    }
    g
}

pub fn random_unitary_haar(rng: &mut ThreadRng, n: usize) -> na::DMatrix<Complex<f64>> {
    /* Random unitary operator of size (n,n) with distribution given by the Haar measure */
    let g = random_ginibre(rng, n);
    let (q, r) = g.qr().unpack();
    let d = r.diagonal().map(|x| x / x.abs());
    let lmda = na::DMatrix::from_diagonal(&d);
    let u = q.adjoint() * lmda;
    u
}

pub fn random_dm_hs(rng: &mut ThreadRng, n: usize) -> na::DMatrix<Complex<f64>> {
    /* Random density matrix of size (n,n) from the Hilbert-Schmidt ensemble */
    let rho = loop {
        // In rare cases a matrix with trace 0 might be returned
        // Thus we need to loop till we make sure that is not the case
        let g = random_ginibre(rng, n);
        let g_dag = g.adjoint();
        let rho = g * g_dag;
        let rho_tr = rho.trace();
        if rho_tr.real() != 0.0 {
            break rho / rho_tr
        }
    };
    rho
}

pub fn random_dm_bures(rng: &mut ThreadRng, n: usize) -> na::DMatrix<Complex<f64>> {
    /* Random density matrix of size (n,n) from the Bures ensemble */
    let rho = loop {
        // In rare cases a matrix with trace 0 might be returned
        // Thus we need to loop till we make sure that is not the case
        let g = random_ginibre(rng, 4);
        let g_dag = g.adjoint();
        let u = random_unitary_haar(rng, 4);
        let u_dag = u.adjoint();
        let i1 = na::DMatrix::<Complex<f64>>::identity(n,n);
        let i2 = na::DMatrix::<Complex<f64>>::identity(n,n);
        let rho = (i1+u)*(g*g_dag)*(i2+u_dag);
        let rho_tr = rho.trace();
        if rho_tr.real() != 0.0 {
            break rho / rho_tr
        }
    };
    rho
}

pub fn random_qubit_pure(rng: &mut ThreadRng) -> na::Matrix2<Complex<f64>> {
    /* Generate a random pure density matrix of a qubit starting from its Bloch representation */
    let (theta,phi) = random_pos_sphere(rng);
    let psi = na::Matrix2x1::new( Complex::new(f64::cos(theta/2.0),0.0), Complex::new(0.0, phi).exp() * Complex::new(f64::sin(theta/2.0),0.0) );
    let rho =  psi * psi.adjoint();
    rho
}

pub fn random_qubit_mixed(rng: &mut ThreadRng) -> na::Matrix2<Complex<f64>> {
    /* Generate a random qubit density matrix from the Ginibre ensemble */
    let rho_dyn = random_dm_hs(rng, 2);
    let rho: na::Matrix2<Complex<f64>> = rho_dyn.fixed_slice::<2, 2>(0, 0).into();
    rho / rho.trace()
}

pub fn random_qubit(rng: &mut ThreadRng) -> na::Matrix2<Complex<f64>> {
    /* Generate a random qubit density matrix */
    let pure: bool = rng.gen();
    if pure {
        return random_qubit_pure(rng);
    }
    else {
        return random_qubit_mixed(rng);
    }
}

pub fn random_bipartite_classical(rng: &mut ThreadRng) -> na::Matrix4<Complex<f64>> {
    /* Generate a random 2 qubit classical state */
    let zero = na::Matrix2::new( na::Complex::new(1.0,0.0), na::Complex::new(0.0,0.0), na::Complex::new(0.0,0.0), na::Complex::new(0.0,0.0) );
    let one = na::Matrix2::new( na::Complex::new(0.0,0.0), na::Complex::new(0.0,0.0), na::Complex::new(0.0,0.0), na::Complex::new(1.0,0.0) );
    let rho1 =  random_qubit(rng);
    let rho2 =  random_qubit(rng);
    let p1 = Uniform::new_inclusive(0.0, 1.0).sample(rng);
    let p2 = 1.0-p1;
    let chi = ( zero.kronecker(&rho1).scale(p1) ) + ( one.kronecker(&rho2).scale(p2) );
    chi
}

pub fn random_bipartite_separable(rng: &mut ThreadRng) -> na::Matrix4<Complex<f64>> {
    /* Generate a random 2 qubit separable state */
    let rho_sep = loop {
        let rho = random_dm_hs(rng, 4).fixed_slice::<4, 4>(0, 0).into();
        if peres_horodecki_separable(&rho) {
            break rho
        }
    };
    rho_sep
}

pub fn correlation_measure_parallel(rho_ab: &na::Matrix4<Complex<f64>>, corr_type: QCorrelation, n_times: usize, n_threads: u8) -> Result<f64, &str> {
    /* Calculate the value of the correlaton measure given by corr_type.
       n_times density matrices with zero correlations of the corresponding type
       will be randomly generated and the minimum distance computed.
       Currently implemented correlation measures include:
         - Geometric Discord
         - Trace Distance Discord
         - Relative Entropy based Discord
         - Entropy of Entanglement
    */
    let rho_ab_in = rho_ab.clone();
    let min_distance: f64 = f64::INFINITY;
    let min_distance_arc = Arc::new(Mutex::new(min_distance));
    let min_distance_ref_watch = Arc::clone(&min_distance_arc);
    let calc_counter: usize = 0;
    let calc_counter_arc = Arc::new(Mutex::new(calc_counter));
    let calc_counter_ref_watch = Arc::clone(&calc_counter_arc);
    let calc_finished = false;
    let calc_finished_arc = Arc::new(Mutex::new(calc_finished));
    let calc_finished_ref_watch = Arc::clone(&calc_finished_arc);
    let mut handles = vec![];
    for thread_id in 0..n_threads {
        let rho_ab_clone = rho_ab_in.clone();
        let min_distance_ref = Arc::clone(&min_distance_arc);
        let calc_counter_ref = Arc::clone(&calc_counter_arc);
        let handle = thread::spawn(move || {
            let mut rng = rand::thread_rng();
            let mult_factor: f64 = match corr_type {
                QCorrelation::Discord(discord_measure) => match discord_measure {
                    DiscordMeasure::GometricDiscord => loop {
                        let chi = random_bipartite_classical(&mut rng);
                        let distance = hilbert_schmidt_distance(&rho_ab_clone, &chi);
                        let mut min_dist = min_distance_ref.lock().unwrap();
                        let mut n_calcs = calc_counter_ref.lock().unwrap();
                        *n_calcs += 1;
                        if *n_calcs <= n_times {
                            if distance < *min_dist {
                                *min_dist = distance;
                                if almost::zero(distance) {
                                    break 1.0
                                }
                            }
                        }
                        else {break 1.0}
                        },
                    DiscordMeasure::TraceDistanceDiscord => loop {
                        let chi = random_bipartite_classical(&mut rng);
                        let distance = trace_distance_herm(&rho_ab_clone, &chi);
                        let mut min_dist = min_distance_ref.lock().unwrap();
                        let mut n_calcs = calc_counter_ref.lock().unwrap();
                        *n_calcs += 1;
                        if *n_calcs <= n_times {
                            if distance < *min_dist {
                                *min_dist = distance;
                                if almost::zero(distance) {
                                    break 0.5
                                }
                            }
                        }
                        else {break 0.5}
                        },
                    DiscordMeasure::RelativeEntropyDiscord => loop {
                        let chi = random_bipartite_classical(&mut rng);
                        let distance = entropy_relative(&rho_ab_clone, &chi, 2.0);
                        let mut min_dist = min_distance_ref.lock().unwrap();
                        let mut n_calcs = calc_counter_ref.lock().unwrap();
                        *n_calcs += 1;
                        if *n_calcs <= n_times {
                            if distance < *min_dist {
                                *min_dist = distance;
                                if almost::zero(distance) {
                                    break 1.0
                                }
                            }
                        }
                        else {break 1.0}
                        }
                },
                QCorrelation::Entanglement(entg_measure) => match entg_measure {
                    EntanglementMeasure::RelativeEntropyOfEntanglement => loop {
                        let sigma = random_bipartite_separable(&mut rng);
                        let distance = entropy_relative(&rho_ab_clone, &sigma, 2.0);
                        let mut min_dist = min_distance_ref.lock().unwrap();
                        let mut n_calcs = calc_counter_ref.lock().unwrap();
                        *n_calcs += 1;
                        if peres_horodecki_separable(&rho_ab_clone) {
                            *min_dist = 0.0;
                            break 1.0
                        }
                        if *n_calcs <= n_times {
                            if distance < *min_dist {
                                *min_dist = distance;
                                if almost::zero(distance) {
                                    break 1.0
                                }
                            }
                        }
                        else {break 1.0}
                        }
                },
            };
            println!("Calculation thread {} exiting.", thread_id);
            mult_factor
            });
        handles.push(handle);
    }
    let watcher_thread = thread::spawn(move || {
        let start = Instant::now();
        loop {
            thread::sleep(Duration::from_secs(2));
            let min_dist = min_distance_ref_watch.lock().unwrap();
            let n_calcs = calc_counter_ref_watch.lock().unwrap();
            let calc_fin = calc_finished_ref_watch.lock().unwrap();
            println!("Time elapsed: {:?}", start.elapsed());
            println!("Number of computations done: {}", *n_calcs);
            println!("Minimum value reached: {}", *min_dist);
            if *calc_fin {
                println!("Watcher thread exiting.");
                break;
            }
        }
    });
    let mut mult_factor: f64 = 1.0;
    for handle in handles {
        mult_factor = handle.join().unwrap();
    }
    {
    let mut cal_fin = calc_finished_arc.lock().unwrap();
    *cal_fin = true;
    }
    watcher_thread.join().unwrap();
    let n_calculations = calc_counter_arc.lock().unwrap();
    if *n_calculations == 0 {
        return Err("Not Implemented")
    }
    let minimum_distance = min_distance_arc.lock().unwrap();
    Ok(mult_factor * minimum_distance.deref())
}

pub fn pauli_matrix(i: usize) -> Result<na::Matrix2<Complex<f64>>,&'static str> {
    /* Return the Pauli matrix σ_i for a given i */
    let pauli_i = match i {
        1 => Ok(na::Matrix2::new(
            na::Complex{re:0.0,im:0.0}, na::Complex{re:1.0,im:0.0},
            na::Complex{re:1.0,im:0.0}, na::Complex{re:0.0,im:0.0},
            )
        ),
        2 => Ok(na::Matrix2::new(
            na::Complex{re:0.0,im:0.0}, na::Complex{re:0.0,im:-1.0},
            na::Complex{re:0.0,im:1.0}, na::Complex{re:0.0,im:0.0},
            )
        ),
        3 => Ok(na::Matrix2::new(
            na::Complex{re:1.0,im:0.0}, na::Complex{re:0.0,im:0.0},
            na::Complex{re:0.0,im:0.0}, na::Complex{re:-1.0,im:0.0},
            )
        ),
        _ => Err("Invalid value of i"),
    };
    pauli_i
}

pub fn werner_state(lmda: f64) -> na::Matrix4<Complex<f64>> {
    /* Generate Werner state W(λ) for the given value of parameter λ */
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

pub fn bell_diagonal_bellbasis(coefficients: [f64; 4]) -> na::Matrix4<Complex<f64>> {
    /* Construct a Bell diagonal state using its form in the Bell-state basis
       for a given array of coefficients: [ λ_1+, λ_1-, λ_2+, λ_2- ]
    */
    let mut bell_diag: na::Matrix4<Complex<f64>> = na::Matrix4::<Complex<f64>>::zeros();
    let zero = na::Matrix2x1::new(
        Complex{re:1.0,im:0.0},
        Complex{re:0.0,im:0.0},
    );
    let one = na::Matrix2x1::new(
        Complex{re:0.0,im:0.0},
        Complex{re:1.0,im:0.0},
    );
    let one_plus = (zero.kronecker(&one)+one.kronecker(&zero)).scale(1.0/(2.0).sqrt());
    let one_minus = (zero.kronecker(&one)-one.kronecker(&zero)).scale(1.0/(2.0).sqrt());
    let two_plus = (zero.kronecker(&zero)+one.kronecker(&one)).scale(1.0/(2.0).sqrt());
    let two_minus = (zero.kronecker(&zero)-one.kronecker(&one)).scale(1.0/(2.0).sqrt());
    let bases = [ one_plus*one_plus.adjoint(), one_minus*one_minus.adjoint(), two_plus*two_plus.adjoint(), two_minus*two_minus.adjoint()];
    for (i, coeff) in coefficients.into_iter().enumerate() {
        bell_diag += bases[i].scale(coeff);
    }
    bell_diag
}

#[allow(non_snake_case)]
pub fn bell_diagonal_bloch(coefficients: [f64; 3]) -> na::Matrix4<Complex<f64>> {
    /* Construct a Bell diagonal state using its Bloch-state representation
       for a given array of coefficients: [ c_1, c_2, c_3 ]
    */
    let mut bell_diag: na::Matrix4<Complex<f64>> = na::Matrix4::<Complex<f64>>::zeros();
    let I_AB = na::Matrix4::<Complex<f64>>::identity();
    bell_diag += I_AB;
    for (i, c_i) in coefficients.into_iter().enumerate() {
        let sigma_i = pauli_matrix(i+1).unwrap();
        bell_diag += (sigma_i.kronecker(&sigma_i)).scale(c_i);
    }
    bell_diag = bell_diag.scale(0.25);
    bell_diag
}
