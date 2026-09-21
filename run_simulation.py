# run_simulation.py
"""
Run simulation study comparing emulator-based and GHK-based estimation
of multinomial probit models.

With --errors gumbel, runs the correlated-Gumbel study instead: emulator-based estimation
compared with softmax-smoothed sample-average simulators (GHK is probit-only). The outside
option is the unit-scale base alternative; see simulation.py for details.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

from simulation import (
    SimulationConfig,
    OptimizationConfig,
    GHKConfig,
    SimulationResults,
    setup_emulator,
    ExactProbitDGP,
    EmulatorDGP,
    run_simulation_study,
    compute_statistics_by_param_type,
    create_results_dataframe_by_param_type,
    print_summary_table,
    generate_latex_tables_by_param_type,
    SmoothedSimConfig,
    GumbelDGP,
    GumbelEmulatorDGP,
    smoothed_method_label,
    DEFAULT_SIM_MEMORY_GB,
    DEFAULT_SIM_CHUNK_ELEMENTS,
)

DEFAULT_SIM_CONFIGS = ["5:500", "5:2000", "20:500", "20:2000"]


def run_comparison_study(
    sim_config: SimulationConfig,
    opt_config: OptimizationConfig,
    ghk_draw_counts: List[int],
    device: torch.device,
    output_dir: Path,
    debug: bool = False,
    run_emulator: bool = True,
    run_ghk: bool = True,
    run_emulator_dgp: bool = True,
    n_bootstrap: int = 1000,
):
    """Run the full comparison study."""
    
    # Count total methods for progress display
    num_emulator_methods = (2 if run_emulator_dgp else 1) if run_emulator else 0
    num_ghk_variants = len(ghk_draw_counts) if run_ghk else 0
    total_methods = num_emulator_methods + num_ghk_variants
    
    is_factor_model = sim_config.covariance_structure == "factor"
    k = sim_config.k
    
    # Print configuration
    print("=" * 70)
    print("MULTINOMIAL PROBIT ESTIMATION: EMULATOR VS GHK COMPARISON")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Sample sizes: {sim_config.sample_sizes}")
    print(f"Number of replications: {sim_config.num_reps}")
    print(f"Number of alternatives (k): {k}")
    print(f"Number of covariates (p): {sim_config.p}")
    print(f"Covariance structure: {sim_config.covariance_structure}")
    print(f"Run Emulator: {run_emulator}")
    if run_emulator:
        print(f"  Run Emulator DGP: {run_emulator_dgp}")
    print(f"Run GHK: {run_ghk}")
    if run_ghk:
        print(f"GHK draw counts: {ghk_draw_counts}")
    print(f"Random seed: {sim_config.seed}")
    print("=" * 70)
    
    # Setup emulator model
    emulator_model = None
    if run_emulator:
        emulator_model = setup_emulator(
            checkpoint_path=sim_config.checkpoint_path,
            device=device
        )
    
    # Setup DGPs
    probit_dgp = ExactProbitDGP()
    emulator_dgp = EmulatorDGP() if run_emulator and run_emulator_dgp else None
    
    # Setup GHK configs (run_simulation_study sets a per-replication GHK seed)
    ghk_configs = [
        GHKConfig(num_draws=num_draws)
        for num_draws in ghk_draw_counts
    ] if run_ghk else []
    
    all_results: List[SimulationResults] = []
    all_stats_by_type: List[Dict[str, Dict[str, Any]]] = []
    
    # Run simulations for each sample size
    for n in sim_config.sample_sizes:
        print(f"\n{'='*70}")
        print(f"SAMPLE SIZE: n = {n:,}")
        print(f"{'='*70}")
        
        method_idx = 1
        emulator_dgp_seed = sim_config.seed
        probit_seed = sim_config.seed + 500000
        
        # Common kwargs for run_simulation_study
        common_kwargs = dict(
            num_reps=sim_config.num_reps,
            n=n,
            p=sim_config.p,
            k=k,
            device=device,
            opt_config=opt_config,
            debug=debug,
            covariance_structure=sim_config.covariance_structure,
        )
        
        if run_emulator:
            # Emulator + Emulator DGP
            if run_emulator_dgp:
                print(f"\n[{method_idx}/{total_methods}] Emulator + Emulator DGP...")
                results = run_simulation_study(
                    dgp=emulator_dgp,
                    estimation_method="emulator",
                    model=emulator_model,
                    seed=emulator_dgp_seed,
                    **common_kwargs,
                )
                all_results.append(results)
                stats_by_type = compute_statistics_by_param_type(results, n_bootstrap=n_bootstrap)
                all_stats_by_type.append(stats_by_type)
                print(f"    RMSE: {stats_by_type['all']['rmse'][0]:.4f}, Coverage: {stats_by_type['all']['coverage'][0]:.1%}")
                method_idx += 1
            
            # Emulator + Probit DGP
            print(f"\n[{method_idx}/{total_methods}] Emulator + Probit DGP...")
            results = run_simulation_study(
                dgp=probit_dgp,
                estimation_method="emulator",
                model=emulator_model,
                seed=probit_seed,
                **common_kwargs,
            )
            all_results.append(results)
            stats_by_type = compute_statistics_by_param_type(results, n_bootstrap=n_bootstrap)
            all_stats_by_type.append(stats_by_type)
            print(f"    RMSE: {stats_by_type['all']['rmse'][0]:.4f}, Coverage: {stats_by_type['all']['coverage'][0]:.1%}")
            method_idx += 1
        
        # GHK methods
        if run_ghk:
            for ghk_config in ghk_configs:
                print(f"\n[{method_idx}/{total_methods}] GHK({ghk_config.num_draws}) + Probit DGP...")
                results = run_simulation_study(
                    dgp=probit_dgp,
                    estimation_method="ghk",
                    ghk_config=ghk_config,
                    seed=probit_seed,
                    **common_kwargs,
                )
                all_results.append(results)
                stats_by_type = compute_statistics_by_param_type(results, n_bootstrap=n_bootstrap)
                all_stats_by_type.append(stats_by_type)
                print(f"    RMSE: {stats_by_type['all']['rmse'][0]:.4f}, Coverage: {stats_by_type['all']['coverage'][0]:.1%}")
                method_idx += 1
    
    # Create DataFrames by parameter type
    dfs_by_type = create_results_dataframe_by_param_type(all_stats_by_type)
    
    # Print summary tables
    default_columns = ['RMSE', 'RMS_Bias', 'SE_Ratio', 'Coverage', 'Time']
    
    print("\n" + "=" * 70)
    if is_factor_model:
        print("ALL PARAMETERS (beta, psi, gamma)")
    else:
        print("ALL PARAMETERS (beta, variance, covariance)")
    print("=" * 70)
    print_summary_table(dfs_by_type['all'])
    
    # Parameter type labels for summary tables
    if is_factor_model:
        param_type_labels = [
            ('beta', 'COEFFICIENTS'), 
            ('psi', 'IDIOSYNCRATIC VARIANCES'),
            ('gamma', 'FACTOR LOADINGS'),
            ('log_psi', 'LOG IDIOSYNCRATIC VARIANCES'),
        ]
    else:
        param_type_labels = [
            ('beta', 'COEFFICIENTS'), 
            ('variance', 'VARIANCE PARAMETERS'),
            ('covariance', 'COVARIANCE PARAMETERS'),
            ('L_diag', 'LOG-DIAGONAL CHOLESKY'), 
            ('L_lower', 'OFF-DIAGONAL CHOLESKY'),
        ]
    
    for param_type, label in param_type_labels:
        if param_type in dfs_by_type:
            print("\n" + "=" * 70)
            print(label)
            print("=" * 70)
            print_summary_table(dfs_by_type[param_type])
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    structure_suffix = "_factor" if is_factor_model else "_dense"
    
    # Save CSVs
    for param_type, df in dfs_by_type.items():
        csv_path = output_dir / f"simulation_results_k{k}_{param_type}{structure_suffix}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults ({param_type}) saved to: {csv_path}")
    
    # Generate and save LaTeX tables
    latex_tables = generate_latex_tables_by_param_type(
        dfs_by_type,
        caption_prefix="Comparison of Emulator and GHK estimation methods for multinomial probit models",
        label_prefix="tab:emulator_vs_ghk",
        k=k,
        columns=default_columns,
    )
    
    for param_type, latex_table in latex_tables.items():
        latex_path = output_dir / f"simulation_table_k{k}_{param_type}{structure_suffix}_{timestamp}.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table ({param_type}) saved to: {latex_path}")
    
    return dfs_by_type, all_results


def run_gumbel_comparison_study(
    sim_config: SimulationConfig,
    opt_config: OptimizationConfig,
    smoothed_configs: List[SmoothedSimConfig],
    device: torch.device,
    output_dir: Path,
    debug: bool = False,
    run_emulator: bool = True,
    run_smoothed: bool = True,
    run_emulator_dgp: bool = True,
    n_bootstrap: int = 1000,
):
    """
    Correlated-Gumbel study (dense covariance): emulator vs softmax-smoothed simulators.

    Mirrors run_comparison_study. The Gumbel RUM uses the same seed offset as the Probit
    DGP there, so covariates and true parameters match the corresponding probit runs.
    """
    if sim_config.covariance_structure != "dense":
        raise ValueError("The Gumbel study supports the dense covariance structure only")
    
    error_distribution = "gumbel"
    k = sim_config.k
    smoothed_configs = smoothed_configs if run_smoothed else []
    
    num_emulator_methods = (2 if run_emulator_dgp else 1) if run_emulator else 0
    total_methods = num_emulator_methods + len(smoothed_configs)
    
    print("=" * 70)
    print("CORRELATED-GUMBEL ESTIMATION: EMULATOR VS SMOOTHED SIMULATOR COMPARISON")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Sample sizes: {sim_config.sample_sizes}")
    print(f"Number of replications: {sim_config.num_reps}")
    print(f"Number of alternatives (k): {k}")
    print(f"Number of covariates (p): {sim_config.p}")
    print(f"Covariance structure: {sim_config.covariance_structure}")
    print("Outside option: unit-scale Gumbel base (Sigma = blockdiag(Omega, 1)), Omega_11 = 1")
    print(f"Run Emulator: {run_emulator}")
    if run_emulator:
        print(f"  Run Emulator DGP: {run_emulator_dgp}")
    print(f"Run smoothed simulators: {bool(smoothed_configs)}")
    for cfg in smoothed_configs:
        print(f"  {smoothed_method_label(cfg.tau, cfg.num_samples)}: tau={cfg.tau:g}, R={cfg.num_samples}, "
              f"memory budget={cfg.memory_gb:g} GB")
    print("GHK: not available for Gumbel errors")
    print(f"Random seed: {sim_config.seed}")
    print("=" * 70)
    
    emulator_model = None
    if run_emulator:
        emulator_model = setup_emulator(
            checkpoint_path=sim_config.checkpoint_path,
            device=device
        )
        if not getattr(emulator_model, 'include_c_star', False):
            raise ValueError(
                f"{sim_config.checkpoint_path} is not a Gumbel emulator (include_c_star=False)"
            )
    
    gumbel_dgp = GumbelDGP()
    emulator_dgp = GumbelEmulatorDGP() if run_emulator and run_emulator_dgp else None
    
    all_results: List[SimulationResults] = []
    all_stats_by_type: List[Dict[str, Dict[str, Any]]] = []
    
    for n in sim_config.sample_sizes:
        print(f"\n{'='*70}")
        print(f"SAMPLE SIZE: n = {n:,}")
        print(f"{'='*70}")
        
        method_idx = 1
        emulator_dgp_seed = sim_config.seed
        rum_seed = sim_config.seed + 500000  # same offset as the Probit DGP
        
        common_kwargs = dict(
            num_reps=sim_config.num_reps,
            n=n,
            p=sim_config.p,
            k=k,
            device=device,
            opt_config=opt_config,
            debug=debug,
            covariance_structure=sim_config.covariance_structure,
            error_distribution=error_distribution,
        )
        
        def record(results: SimulationResults):
            all_results.append(results)
            stats_by_type = compute_statistics_by_param_type(results, n_bootstrap=n_bootstrap)
            all_stats_by_type.append(stats_by_type)
            print(f"    RMSE: {stats_by_type['all']['rmse'][0]:.4f}, Coverage: {stats_by_type['all']['coverage'][0]:.1%}")
        
        if run_emulator:
            if run_emulator_dgp:
                print(f"\n[{method_idx}/{total_methods}] Emulator + Emulator DGP...")
                record(run_simulation_study(
                    dgp=emulator_dgp,
                    estimation_method="emulator",
                    model=emulator_model,
                    seed=emulator_dgp_seed,
                    **common_kwargs,
                ))
                method_idx += 1
            
            print(f"\n[{method_idx}/{total_methods}] Emulator + Gumbel DGP...")
            record(run_simulation_study(
                dgp=gumbel_dgp,
                estimation_method="emulator",
                model=emulator_model,
                seed=rum_seed,
                **common_kwargs,
            ))
            method_idx += 1
        
        for smoothed_config in smoothed_configs:
            label = smoothed_method_label(smoothed_config.tau, smoothed_config.num_samples)
            print(f"\n[{method_idx}/{total_methods}] {label} + Gumbel DGP...")
            record(run_simulation_study(
                dgp=gumbel_dgp,
                estimation_method="smoothed",
                smoothed_config=smoothed_config,
                seed=rum_seed,
                **common_kwargs,
            ))
            method_idx += 1
    
    dfs_by_type = create_results_dataframe_by_param_type(all_stats_by_type)
    
    default_columns = ['RMSE', 'RMS_Bias', 'SE_Ratio', 'Coverage', 'Time']
    
    print("\n" + "=" * 70)
    print("ALL PARAMETERS (beta, variance, covariance)")
    print("=" * 70)
    print_summary_table(dfs_by_type['all'])
    
    param_type_labels = [
        ('beta', 'COEFFICIENTS'), 
        ('variance', 'VARIANCE PARAMETERS'),
        ('covariance', 'COVARIANCE PARAMETERS'),
        ('L_diag', 'LOG-DIAGONAL CHOLESKY'), 
        ('L_lower', 'OFF-DIAGONAL CHOLESKY'),
    ]
    for param_type, label in param_type_labels:
        if param_type in dfs_by_type:
            print("\n" + "=" * 70)
            print(label)
            print("=" * 70)
            print_summary_table(dfs_by_type[param_type])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    structure_suffix = "_dense_gumbel"
    
    for param_type, df in dfs_by_type.items():
        csv_path = output_dir / f"simulation_results_k{k}_{param_type}{structure_suffix}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults ({param_type}) saved to: {csv_path}")
    
    latex_tables = generate_latex_tables_by_param_type(
        dfs_by_type,
        caption_prefix=(
            "Comparison of Emulator and softmax-smoothed simulator (Sim($\\tau$, $R$)) "
            "estimation methods for correlated-Gumbel models"
        ),
        label_prefix="tab:emulator_vs_sim_gumbel",
        k=k,
        columns=default_columns,
    )
    
    for param_type, latex_table in latex_tables.items():
        latex_path = output_dir / f"simulation_table_k{k}_{param_type}{structure_suffix}_{timestamp}.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table ({param_type}) saved to: {latex_path}")
    
    return dfs_by_type, all_results


def parse_sim_configs(
    specs: List[str],
    memory_gb: float,
    chunk_elements: int,
    use_checkpoint: Optional[bool],
) -> List[SmoothedSimConfig]:
    """Parse 'tau:R' strings (e.g. '5:500') into SmoothedSimConfig objects."""
    configs = []
    for spec in specs:
        try:
            tau_str, r_str = spec.split(":")
            tau, num_samples = float(tau_str), int(r_str)
        except ValueError:
            raise ValueError(f"Invalid --sim-configs entry {spec!r}; expected tau:R, e.g. 5:500")
        if tau <= 0 or num_samples < 1:
            raise ValueError(f"Invalid --sim-configs entry {spec!r}; need tau > 0 and R >= 1")
        configs.append(SmoothedSimConfig(
            tau=tau,
            num_samples=num_samples,
            memory_gb=memory_gb,
            chunk_elements=chunk_elements,
            use_checkpoint=use_checkpoint,
        ))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Simulation study: emulator vs GHK (probit) or vs smoothed simulators (--errors gumbel)"
    )
    
    # Simulation parameters
    parser.add_argument("--num-reps", type=int, default=100)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    
    # Method parameters
    parser.add_argument("--ghk-draws", type=int, nargs="+", default=[10, 50, 250])
    parser.add_argument("--checkpoint", type=str, default="./model-checkpoints/checkpoint_emulator_3.pt",
                        help="Path to emulator checkpoint")
    
    # Optimization parameters
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap samples for standard errors")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./simulation-results")
    parser.add_argument("--debug", action="store_true")
    
    # Method selection flags
    parser.add_argument("--no-emulator", action="store_true", help="Skip emulator methods")
    parser.add_argument("--no-ghk", action="store_true", help="Skip GHK methods")
    parser.add_argument("--no-emulator-dgp", action="store_true", 
                        help="Skip emulator DGP (only run emulator estimation with probit DGP)")
    
    # Covariance structure flags
    parser.add_argument("--factor", action="store_true", 
                        help="Use factor covariance structure instead of dense")
    
    # Error distribution and smoothed-simulator flags (Gumbel study)
    parser.add_argument("--errors", choices=["probit", "gumbel"], default="probit",
                        help="Error distribution. 'gumbel' runs the correlated-Gumbel study "
                             "(dense only) with softmax-smoothed simulators instead of GHK")
    parser.add_argument("--sim-configs", type=str, nargs="+", default=DEFAULT_SIM_CONFIGS,
                        help="Smoothed simulators as tau:R pairs (Gumbel only)")
    parser.add_argument("--sim-memory-gb", type=float, default=DEFAULT_SIM_MEMORY_GB,
                        help="Memory budget per smoothed-simulator fit (sets pool size and checkpointing)")
    parser.add_argument("--sim-chunk-elements", type=int, default=DEFAULT_SIM_CHUNK_ELEMENTS,
                        help="Rows x draws x K per smoothed-simulator chunk")
    parser.add_argument("--sim-checkpoint", choices=["auto", "on", "off"], default="auto",
                        help="Checkpoint smoothed-simulator chunks: from the memory budget (auto), always, or never")
    parser.add_argument("--no-smoothed", action="store_true",
                        help="Skip the smoothed simulators (Gumbel only)")
    
    args = parser.parse_args()
    
    if args.errors == "gumbel":
        if args.factor:
            parser.error("--errors gumbel supports only the dense covariance structure (drop --factor)")
        
        sim_config = SimulationConfig(
            sample_sizes=args.sample_sizes,
            num_reps=args.num_reps,
            p=args.p,
            k=args.k,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            covariance_structure="dense",
            error_distribution="gumbel",
        )
        
        opt_config = OptimizationConfig(
            max_iter=args.max_iter,
            loss_tol_base=1e-8,
            grad_tol=1e-9,
            history_size=50,
        )
        
        use_checkpoint = {"auto": None, "on": True, "off": False}[args.sim_checkpoint]
        try:
            smoothed_configs = parse_sim_configs(
                args.sim_configs, args.sim_memory_gb, args.sim_chunk_elements, use_checkpoint
            )
        except ValueError as exc:
            parser.error(str(exc))
        
        run_gumbel_comparison_study(
            sim_config=sim_config,
            opt_config=opt_config,
            smoothed_configs=smoothed_configs,
            device=torch.device("cpu"),
            output_dir=Path(args.output_dir),
            debug=args.debug,
            run_emulator=not args.no_emulator,
            run_smoothed=not args.no_smoothed,
            run_emulator_dgp=not args.no_emulator_dgp,
            n_bootstrap=args.n_bootstrap,
        )
        return
    
    sim_config = SimulationConfig(
        sample_sizes=args.sample_sizes,
        num_reps=args.num_reps,
        p=args.p,
        k=args.k,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        covariance_structure="factor" if args.factor else "dense"
    )
    
    opt_config = OptimizationConfig(
        max_iter=args.max_iter,
        loss_tol_base=1e-8,
        grad_tol=1e-9,
        history_size=50,
    )
    
    run_comparison_study(
        sim_config=sim_config,
        opt_config=opt_config,
        ghk_draw_counts=args.ghk_draws,
        device=torch.device("cpu"),
        output_dir=Path(args.output_dir),
        debug=args.debug,
        run_emulator=not args.no_emulator,
        run_ghk=not args.no_ghk,
        run_emulator_dgp=not args.no_emulator_dgp,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()