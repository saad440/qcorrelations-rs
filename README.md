# qcorrelations-rs
Computational library to analyse the dynamics of quantum correlations, written in Rust.  
It currently focuses on measures computed by distance minimization over some set of states for which an analytical solution may not necessarily be available. Multiple threads are used to accelerate the minimization.

Warning: This is a work in progress and not thoroughly tested. There might be glitches and behaviour might drastically change as it is updated.

## Currently available functions
### General Measures and Operations
- Von Neumann Entropy
- Relative Entropy
- Hilbert-Schmidt distance
- Trace Distance (for Hermitian matrices)
- Matrix logarithm (for diagonalizable matrices)
- Partial transpose (for bipartite states)
- Random matrix from the Ginibre ensemble
### Correlation Witnesses
- Entanglement: Peres–Horodecki criterion
### Correlation Measures
- Relative Entropy of Entanglement
- Relative Entropy based Discord
- Geometric Discord
- Trace Distance Discord
### Quantum state / operator constructors
- Random pure state
- Random mixed state
- Random unitary operator
- Random classical state
- Random separable state
- Random entangled state
- Werner states
- Bell diagonal states
